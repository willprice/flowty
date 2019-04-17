# cython: language_level = 3
from libcpp cimport bool
from libcpp.memory cimport shared_ptr
from .c_core cimport Mat, InputArray, InputOutputArray


cdef extern from "opencv2/video/tracking.hpp" namespace "cv":
    cdef cppclass DenseOpticalFlow:
        void calc(InputArray i0, InputArray i1, InputOutputArray flow)
        void collectGarbage()

cdef extern from "opencv2/optflow.hpp" namespace "cv::optflow":
    cdef cppclass DualTVL1OpticalFlow(DenseOpticalFlow):
        @staticmethod
        shared_ptr[DualTVL1OpticalFlow] create(double, double, double, int, int, double, int, int, double, double, int, bool) except+
        double getEpsilon()
        double getGamma()
        double getLambda()
        double getTau()
        double getTheta()
        int getMedianFiltering()
        int getScalesNumber()
        double getScaleStep()
        int getInnerIterations()
        int getOuterIterations()
        bool getUseInitialFlow()
        int getWarpingsNumber()

        void setEpsilon(double)
        void setGamma(double)
        void setInnerIterations(int)
        void setOuterIterations(int)
        void setLambda(double)
        void setMedianFiltering(int)
        void setScalesNumber(int)
        void setScaleStep(double)
        void setTau(double)
        void setTheta(double)
        void setUseInitialFlow(bool)
        void setWarpingsNumber(int)
