import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker
import pickle

MARKER_SIZE = 16
MARKER_SIZE_SMALL = 1
LINE_WIDTH = 3
LINE_WIDTH_SMALL = 1
FONT_SIZE = 20
FONT_SIZE_SMALL = 5
LEGEND_FONT_SIZE = 14
LEGEND_FONT_SIZE_SMALL = 5


class DistParser(object):
    """
    Parse pcl files, and generate plots from them.
    """

    # Set the parameters of the plot (sizes of fonts, legend, ticks etc.).
    # mfc='none' makes the markers empty.

    def __init__(self):
        """
        Initialize a pcl_file_parser, used to parse result files, and generate plots.
        """
        # List of algorithms' names, used in the plots' legend, for the dist' case
        self.labelOfMode = {}

        # The colors used for each alg's plot, in the dist' case
        self.colorOfFP = {'uniform': 'blue',
                          'Gauss': 'red',
                          'student': 'green'}

        # The markers used for each alg', in the dist' case
        self.points = []

    def rdPcl(self):
        """
        Given a RdRmse.pcl, read all the data it contains into self.points
        """
        pclFile = open('res/pcl_files/{}.pcl'.format("distributions"), 'rb')

        while True:
            try:
                self.points.append(pickle.load(pclFile))
            except EOFError:
                break
        print(self.points)

    def distPlot(self):
        """
        Generate a plot showing the Normalized_RMSE vs. width.
        """
        for data in self.points:
           floatinput=data['floatInput']
           for key, value in data.items():
               if key!='floatInput':
                  plt.plot(floatinput, value, color=self.colorOfFP[key], label=key)
        plt.xlabel('floatInput')
        plt.ylabel('Probability')
        plt.title('Different Distributions')
        plt.legend()
        plt.savefig('res/result.pdf'.format("result"), bbox_inches='tight')


if __name__ == '__main__':
    fp = DistParser()
    fp.rdPcl()
    fp.distPlot()
