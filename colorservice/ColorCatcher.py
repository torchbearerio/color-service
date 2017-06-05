from pythoncore import Task, Constants
from pythoncore.Model import TorchbearerDB, Hit
from pythoncore.AWS import AWSClient
from PIL import Image
import math
import numpy as np
from scipy.spatial import distance
from colorthief import ColorThief
import cv2
import webcolors
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie1976
from colormath.color_objects import LabColor, sRGBColor, HSLColor
import Hues
import traceback
import io


class ColorCatcher(Task.Task):
    def __init__(self, ep_id, hit_id, task_token):
        super(ColorCatcher, self).__init__(ep_id, hit_id, task_token)

    def run(self):
        # Create DB session
        session = TorchbearerDB.Session()

        try:
            # Get landmarks for hit
            hit = session.query(Hit.Hit).filter_by(hit_id=self.hit_id).one()
            landmarks = hit.candidate_landmarks

            for landmark in landmarks:
                # Load transparent cropped landmark image (as file)
                image = self._get_landmark_image(landmark.landmark_id)

                colors = self.get_dominant_colors(image)

                landmark.set_colors(colors)

            # Commit DB inserts/updates
            session.commit()

            # Send success!
            self.send_success()

        except Exception, e:
            traceback.print_exc()
            session.rollback()
            self.send_failure('COLOR SERVICE ERROR', 'Unable to complete the color description task.')

    @staticmethod
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

    # Note: Returns the image as a binary file, directly from S3. Does NOT convert to np array
    @staticmethod
    def _get_landmark_image(landmark_id):
        client = AWSClient.get_client('s3')
        response = client.get_object(
            Bucket=Constants.S3_BUCKETS['TRANSPARENT_CROPPED_IMAGES'],
            Key="{0}.png".format(landmark_id)
        )
        image = io.BytesIO(response.get("Body").read())
        return image
        # img.show()

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

        return Hues.get_color_name_from_hue(hsl_triplet.hsl_h)

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


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # img = Image.open('test.png')
    # img.show()
    cc = ColorCatcher(1, 6, '1')
    cc.run()
