from libcpp cimport bool
from libcpp.vector cimport vector
from c_core cimport Ptr, InputArray, InputOutputArray


cdef extern from "opencv2/cudaoptflow.hpp" namespace "cv":
    cdef cppclass DenseOpticalFlow:
        void calc(InputArray, InputArray, InputOutputArray flow)
        void collectGarbage()

    cdef cppclass OpticalFlowDual_TVL1(DenseOpticalFlow):

        @staticmethod
        Ptr[OpticalFlowDual_TVL1] create(double, double, double, int, int, double, int, double, double, bool)

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
