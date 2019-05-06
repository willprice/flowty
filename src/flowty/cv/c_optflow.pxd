# cython: language_level = 3
from libcpp cimport bool
from .c_core cimport Ptr, InputArray, InputOutputArray, String, Mat


cdef extern from "opencv2/video/tracking.hpp" namespace "cv" nogil:
    cdef cppclass DenseOpticalFlow:
        void calc(InputArray i0, InputArray i1, InputOutputArray flow) except +
        void collectGarbage() except +

    cdef cppclass FarnebackOpticalFlow(DenseOpticalFlow):
        @staticmethod
        Ptr[FarnebackOpticalFlow] create(int, double, bool, int, int, int, double) except +

        bool getFastPyramids()
        int getFlags()
        int getNumIters()
        int getNumLevels()
        int getPolyN()
        double getPolySigma()
        double getPyrScale()
        int getWinSize()

        void setFastPyramids(bool)
        void setFlags(int)
        void setNumIters(int)
        void setNumLevels(int)
        void setPolyN(int)
        void setPolySigma(double)
        void setPyrScale(double)
        void setWinSize(int)

    cdef cppclass DISOpticalFlow(DenseOpticalFlow):
        @staticmethod
        Ptr[DISOpticalFlow] create(int) except +

        int getFinestScale()
        int getGradientDescentIterations()
        int getPatchSize()
        int getPatchStride()
        bool getUseMeanNormalization()
        bool getUseSpatialPropagation()
        float getVariationalRefinementAlpha()
        float getVariationalRefinementDelta()
        float getVariationalRefinementGamma()
        int getVariationalRefinementIterations()

        void setFinestScale(int)
        void setGradientDescentIterations(int)
        void setPatchSize(int)
        void setPatchStride(int)
        void setUseMeanNormalization(bool)
        void setUseSpatialPropagation(bool)
        void setVariationalRefinementAlpha(float)
        void setVariationalRefinementDelta(float)
        void setVariationalRefinementGamma(float)
        void setVariationalRefinementIterations(int)

    cdef cppclass VariationalRefinement(DenseOpticalFlow):
        @staticmethod
        Ptr[VariationalRefinement] create() except +

        float getAlpha()
        float getDelta()
        int getFixedPointIterations()
        float getGamma()
        float getOmega()
        int getSorIterations()

        void setAlpha(float)
        void setDelta(float)
        void setFixedPointIterations(int)
        void setGamma(float)
        void setOmega(float)
        void setSorIterations(int)

    bool writeOpticalFlow(String&, InputArray)
    Mat readOpticalFlow(String&)

cdef extern from "opencv2/video/tracking.hpp" namespace "cv::DISOpticalFlow":
        enum:
            PRESET_ULTRAFAST
            PRESET_FAST
            PRESET_MEDIUM

cdef extern from "opencv2/optflow.hpp" namespace "cv::optflow" nogil:
    cdef cppclass DualTVL1OpticalFlow(DenseOpticalFlow):
        @staticmethod
        Ptr[DualTVL1OpticalFlow] create(double, double, double, int, int, double, int,
                                      int, double, double, int, bool) except+
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

    cdef cppclass DenseRLOFOpticalFlow(DenseOpticalFlow):
        @staticmethod
        Ptr[DenseRLOFOpticalFlow] create() except +