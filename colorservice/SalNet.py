import Algorithmia
from Algorithmia.algo_response import AlgoException
from pythoncore import Vault, Task

# Build Algorithmia client
key = Vault.get_key('algorithmia-key')
client = Algorithmia.client(key)


class SalNet (Task.Task):

    def __init__(self, ep_id, task_token):
        super(SalNet, self).__init__(ep_id, task_token)

    def __run_salnet(self):
        # Pass in file and pass in args required from the algorithm FpGrowth
        params = {
            "image": 's3+S3://torchbearer-sv-images/{0}.jpg'.format(self.ep_id),
            "saliencyLocation": 's3+S3://torchbearer-saliency-maps/{0}.json'.format(self.ep_id)
        }

        algo = client.algo('deeplearning/SalNet/0.2.0')

        # Run SalNet, then return
        try:
            response = algo.pipe(params)
            self.send_success()

        except AlgoException:
            self.send_failure('SALNET_ERROR', response.error.message)

    def run(self):
        self.__run_salnet()

if __name__ == '__main__':
    sn = SalNet(21,"sdfasdf")
    sn.run()
