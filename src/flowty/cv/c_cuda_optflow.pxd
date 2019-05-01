from libcpp cimport bool
from libcpp.vector cimport vector
from .c_core cimport Ptr, InputArray, InputOutputArray, Size


cdef extern from "opencv2/cudaoptflow.hpp" namespace "cv::cuda" nogil:
    cdef cppclass DenseOpticalFlow:
        void calc(InputArray, InputArray, InputOutputArray flow) except +
        void collectGarbage() except +

    cdef cppclass OpticalFlowDual_TVL1(DenseOpticalFlow):

        @staticmethod
        Ptr[OpticalFlowDual_TVL1] create(double, double, double, int,
                                         int, double, int, double,
                                         double, bool) except +

        double getEpsilon()
        double getGamma()
        double getLambda()
        int getNumIterations()
        int getNumScales()
        int getNumWarps()
        double getScaleStep()
        double getTau()
        double getTheta()
        bool getUseInitialFlow()

        void setEpsilon(double)
        void setGamma(double)
        void setLambda(double)
        void setNumIterations(int)
        void setNumScales(int)
        void setNumWarps(int)
        void setScaleStep(double)
        void setTau(double)
        void setTheta(double)
        void setUseInitialFlow(bool)

    cdef cppclass BroxOpticalFlow(DenseOpticalFlow):
        @staticmethod
        Ptr[BroxOpticalFlow] create(double, double, double, int, int, int) except +

        # alpha
        double getFlowSmoothness()
        # gamma
        double getGradientConstancyImportance()
        double getPyramidScaleFactor()
        int getInnerIterations()
        int getOuterIterations()
        int getSolverIterations()

        void setFlowSmoothness(double)
        void setGradientConstancyImportance(double)
        void setPyramidScaleFactor(double)
        void setInnerIterations(int)
        void setOuterIterations(int)
        void setSolverIterations(int)

    cdef cppclass DensePyrLKOpticalFlow(DenseOpticalFlow):
        @staticmethod
        Ptr[DensePyrLKOpticalFlow] create(Size, int, int, bool)

        int getMaxLevel()
        int getNumIters()
        int getUseInitialFlow()
        Size getWinSize()

        void setMaxLevel(int)
        void setNumIters(int)
        void setUseInitialFlow(bool)
        void setWinSize(Size)

    cdef cppclass FarnebackOpticalFlow(DenseOpticalFlow):
        @staticmethod
        Ptr[FarnebackOpticalFlow] create(int, double, bool, int, int, int, double)
