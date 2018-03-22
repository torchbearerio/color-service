from pythoncore import Task, Constants
from pythoncore.Model import TorchbearerDB, Hit
from pythoncore.AWS import AWSClient
from PIL import Image
import math
import numpy as np
from colorthief import ColorThief
import cv2
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie1976
from colormath.color_objects import LabColor, sRGBColor, HSLColor
import Hues
import traceback
import io
import os


class ColorCatcher(Task.Task):
    def __init__(self, ep_id, hit_id, task_token):
        super(ColorCatcher, self).__init__(ep_id, hit_id, task_token)
        self.streetview_images = dict()

    def run(self):
        # Create DB session
        session = TorchbearerDB.Session()

        try:
            # Get landmarks for hit and set task start time
            hit = session.query(Hit.Hit).filter_by(hit_id=self.hit_id).one()
            hit.set_start_time_for_task("color")
            landmarks = hit.candidate_landmarks

            for landmark in landmarks:
                colors = []

                if not self.streetview_images.get(landmark.position):
                    self.streetview_images[landmark.position] = self._get_streetview_image(landmark.position)

                # Load transparent cropped landmark image (as file)
                image = self._get_landmark_image(landmark)

                # Make sure image is not entirely transparent (meaning we couldn't find a foreground object)
                if image and self.image_is_valid(Image.open(image), quality=10):
                    colors = self.get_dominant_colors(image)

                landmark.set_colors(colors)

            hit.set_end_time_for_task("color")

            # Commit DB inserts/updates
            session.commit()

            # Send success!
            self.send_success()

        except Exception, e:
            traceback.print_exc()
            session.rollback()
            self.send_failure('COLOR SERVICE ERROR', e.message)

        finally:
            session.close()

    @staticmethod
    # Image is a BytesIO file-like object. Color-Thief will open it with PIL.
    def get_dominant_colors(image):
        # Find best value of n between (1, 3)
        ct = ColorThief(image)
        best_clustering = None
        best_se = float("inf")

        colors = ct.get_palette(5, 10)

        # Add alpha channel to colors for comparability (full opacity)
        # colors = map(lambda color: color + (255,), colors)

        for n in range(1, 4):
            se = ColorCatcher.calculate_se(image, colors[:n])

            # Penalize SSE for number of clusters
            se *= math.sqrt(n)

            if se < best_se:
                best_se = se
                best_clustering = colors[:n]
            else:
                break

        color_names = set(map(lambda color: ColorCatcher.get_color_name(color), best_clustering))

        return color_names

    def _get_streetview_image(self, position):
        client = AWSClient.get_client('s3')
        response = client.get_object(
            Bucket=Constants.S3_BUCKETS['STREETVIEW_IMAGES'],
            Key="{}_{}.jpg".format(self.hit_id, position)
        )
        return Image.open(response['Body'])

    # Note: Returns the image as a binary file, directly from S3. Does NOT convert to np array
    def _get_landmark_image(self, landmark):
        img = self.streetview_images[landmark.position]

        rect = landmark.get_rect()

        if not rect:
            return None

        rect = (rect['x1'], rect['y1'], rect['x2'], rect['y2'])

        # Perform image segmentation to extract foreground object
        cv_img = np.asarray(img, np.uint8)
        # Convert to BGR
        cv_img = cv_img[:, :, ::-1].copy()

        mask = np.zeros(cv_img.shape[:2], np.uint8)
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)

        cv2.grabCut(cv_img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

        # If mask==2 or mask== 1, mask2 get 0, other wise it gets 1 as 'uint8' type.
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

        # Project image into RGBA space
        # cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2BGRA)
        # Convert from BGRA to RGBA
        cv_img = cv_img[:, :, ::-1].copy()
        cv_img = np.concatenate((cv_img, np.full((cv_img.shape[0], cv_img.shape[1], 1), 255)), axis=2)

        # Set alpha based on segment_mask.
        # Set red channel to 100% for viewers that don't support alpha channel.
        cv_img[mask2 == 0] = [255, 0, 0, 0]

        # Convert cv2 img back to PIL img
        cv_img = np.uint8(cv_img)
        img = Image.fromarray(cv_img)

        # Crop cut image down to landmark rect
        img = img.crop(rect)

        # Create file-like object for passing to ColorThief
        img_file = io.BytesIO()
        img.save(img_file, 'PNG')
        img_file.seek(0)
        return img_file

    @staticmethod
    def calculate_se(image, colors):
        # Make sure cursor is at beginning of image file
        image.seek(0)

        pil_img = Image.open(image)
        img_array = np.array(pil_img)

        # Resize it
        h, w, _ = img_array.shape
        w_new = int(100 * w / max(w, h))
        h_new = int(100 * h / max(w, h))
        img_array = cv2.resize(img_array, (w_new, h_new))

        # Filter pixels transparent pixels, reshape to single-row image
        img_array = img_array[img_array[..., 3] == 255]
        img_array = img_array.reshape(1, -1, img_array.shape[-1])

        # Convert image to CIE LAB color space
        # img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2Lab)

        # Convert colors array to LAB
        colors = map(lambda color: convert_color(sRGBColor(color[0], color[1], color[2], is_upscaled=True),
                                                 LabColor), colors)

        se = 0

        for i in np.ndindex(img_array.shape[:2]):
            p = img_array[i]
            p_color = convert_color(sRGBColor(p[0], p[1], p[2], is_upscaled=True),
                                    LabColor)
            delta_e_arr = map(lambda c: delta_e_cie1976(p_color, c), colors)
            se += min(delta_e_arr)

        return se

    @staticmethod
    def get_color_name(rgb_triplet):
        hsl_triplet = convert_color(sRGBColor(rgb_triplet[0], rgb_triplet[1], rgb_triplet[2], is_upscaled=True),
                                    HSLColor)

        return Hues.get_color_name_from_hsl(*hsl_triplet.get_value_tuple())

        # min_colours = {}
        # hsl_triplet.hsl_s = -0.5
        # hsl_triplet.hsl_l = 128
        # lab_triplet = convert_color(hsl_triplet, LabColor)
        #
        # rgb_test = convert_color(hsl_triplet, sRGBColor)
        #
        # for key, name in webcolors.html4_hex_to_names.items():
        #     candidate_rgb = webcolors.hex_to_rgb(key)
        #     candidate_lab = convert_color(sRGBColor(candidate_rgb[0], candidate_rgb[1], candidate_rgb[2]),
        #                                   LabColor)
        #
        #     d = delta_e_cie1976(lab_triplet, candidate_lab)
        #     min_colours[d] = name
        # return min_colours[min(min_colours.keys())]

    @staticmethod
    def image_is_valid(image, quality=10):
        width, height = image.size
        pixels = image.getdata()
        pixel_count = width * height
        for i in range(0, pixel_count, quality):
            r, g, b, a = pixels[i]
            # If pixel is mostly opaque and not white
            if a >= 125:
                if not (r > 250 and g > 250 and b > 250):
                    return True
        return False


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # img = Image.open('test.png')
    # img.show()
    cc = ColorCatcher(191, 455, '1')
    cc.run()
