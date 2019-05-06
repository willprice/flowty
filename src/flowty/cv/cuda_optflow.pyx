# cython: language_level = 3

from libcpp cimport bool
from cython.operator cimport dereference as deref
from ..cv.c_cuda_optflow cimport OpticalFlowDual_TVL1, BroxOpticalFlow, \
    DensePyrLKOpticalFlow, FarnebackOpticalFlow
from ..cv.core cimport Mat
from ..cv.c_core cimport Mat as c_Mat, Size as c_Size
from ..cv.c_core cimport InputArray, OutputArray, InputOutputArray, Ptr, CV_32FC1, \
    CV_32FC2
from ..cv.c_imgproc cimport cvtColor, ColorConversionCodes
from ..cv.c_cuda cimport GpuMat as c_GpuMat


cdef class CudaTvL1OpticalFlow:
    cdef Ptr[OpticalFlowDual_TVL1] alg
    cdef c_GpuMat reference_gpu, target_gpu, flow_gpu
    cdef c_Mat reference, target, flow

    def __cinit__(self,
                  double tau=0.25,
                  double lambda_=0.15,
                  double theta=0.3,
                  double epsilon=0.01,
                  double gamma=0.0,
                  double scale_step=0.8,
                  int scale_count=5,
                  int warp_count=5,
                  int iterations=300,
                  bool use_initial_flow=False):
        self.alg = OpticalFlowDual_TVL1.create(tau, lambda_, theta, scale_count, warp_count, epsilon,
            iterations, scale_step, gamma, use_initial_flow)

    def __call__(self, Mat reference, Mat target) -> Mat:
        cvtColor(<InputArray>reference.c_mat,
                 <OutputArray>self.reference,
                 ColorConversionCodes.COLOR_BGR2GRAY)
        cvtColor(<InputArray>target.c_mat,
                 <OutputArray>self.target,
                 ColorConversionCodes.COLOR_BGR2GRAY)
        if self.flow_gpu.empty():
             self.flow_gpu = c_GpuMat(self.reference.rows,
                                      self.reference.cols,
                                      CV_32FC2)
        with nogil:
            self.target_gpu.upload(<InputArray> self.target)
            self.reference_gpu.upload(<InputArray> self.reference)
            deref(self.alg).calc(<InputArray>self.reference_gpu,
                                 <InputArray>self.target_gpu,
                                 <InputOutputArray>self.flow_gpu)
        flow = Mat()
        self.flow_gpu.download(<OutputArray> flow.c_mat)
        return flow

    @property
    def tau(self):
        return deref(self.alg).getTau()

    @tau.setter
    def tau(self, tau):
        deref(self.alg).setTau(tau)

    @property
    def lambda_(self):
        return deref(self.alg).getLambda()

    @lambda_.setter
    def lambda_(self, lambda_):
        deref(self.alg).setLambda(lambda_)

    @property
    def theta(self):
        return deref(self.alg).getTheta()

    @theta.setter
    def theta(self, theta):
        deref(self.alg).setTheta(theta)

    @property
    def epsilon(self):
        return deref(self.alg).getEpsilon()

    @epsilon.setter
    def epsilon(self, epsilon):
        deref(self.alg).setEpsilon(epsilon)

    @property
    def gamma(self):
        return deref(self.alg).getGamma()

    @gamma.setter
    def gamma(self, gamma):
        deref(self.alg).setGamma(gamma)

    @property
    def scale_step(self):
        return deref(self.alg).getScaleStep()

    @property
    def scale_count(self):
        return deref(self.alg).getNumScales()

    @property
    def warp_count(self):
        return deref(self.alg).getNumWarps()

    @property
    def iterations(self):
        return deref(self.alg).getNumIterations()

    @property
    def use_initial_flow(self):
        return deref(self.alg).getUseInitialFlow()

    def __repr__(self):
        return (self.__class__.__name__ +
            ("("
             "tau={tau}, lambda_={lambda_}, theta={theta}, epsilon={epsilon}, gamma={gamma}, "
             "scale_step={scale_step}, scale_count={scale_count}, warp_count={warp_count}, "
             "iterations={iterations}, use_initial_flow={use_initial_flow}"
             ")"
             ).format(
                tau=self.tau,
                lambda_=self.lambda_,
                theta=self.theta,
                epsilon=self.epsilon,
                gamma=self.gamma,
                scale_count=self.scale_count,
                warp_count=self.warp_count,
                iterations=self.iterations,
                scale_step=self.scale_step,
                use_initial_flow=self.use_initial_flow
            ))

cdef class CudaBroxOpticalFlow:
    cdef Ptr[BroxOpticalFlow] alg
    cdef c_GpuMat reference_gpu, target_gpu, flow_gpu
    cdef c_Mat reference, target, flow, reference_float, target_float

    def __cinit__(self,
                  double alpha=0.197,
                  double gamma=50.0,
                  double scale_factor=0.8,
                  int inner_iterations=5,
                  int outer_iterations=150,
                  int solver_iterations=10,
                  ):
        self.alg = BroxOpticalFlow.create(alpha, gamma, scale_factor, inner_iterations,
                                          outer_iterations, solver_iterations)

    def __call__(self, Mat reference, Mat target) -> Mat:
        cvtColor(<InputArray>reference.c_mat,
                 <OutputArray>self.reference,
                 ColorConversionCodes.COLOR_BGR2GRAY)
        cvtColor(<InputArray>target.c_mat,
                 <OutputArray>self.target,
                 ColorConversionCodes.COLOR_BGR2GRAY)
        self.target.convertTo(<OutputArray> self.target_float, CV_32FC1, 1.0 / 255.0)
        self.reference.convertTo(<OutputArray> self.reference_float, CV_32FC1,
                                 1.0 / 255.0)

        with nogil:
            if self.flow_gpu.empty():
                self.flow_gpu = c_GpuMat(self.reference.rows,
                                         self.reference.cols,
                                         CV_32FC2)
            self.target_gpu.upload(<InputArray> self.target_float)
            self.reference_gpu.upload(<InputArray> self.reference_float)


            deref(self.alg).calc(<InputArray>self.reference_gpu,
                                 <InputArray>self.target_gpu,
                                 <InputOutputArray>self.flow_gpu)
        flow = Mat()
        self.flow_gpu.download(<OutputArray> flow.c_mat)
        return flow

    @property
    def inner_iterations(self):
        return deref(self.alg).getInnerIterations()

    @property
    def outer_iterations(self):
        return deref(self.alg).getOuterIterations()

    @property
    def solver_iterations(self):
        return deref(self.alg).getSolverIterations()

    @property
    def scale_factor(self):
        return deref(self.alg).getPyramidScaleFactor()

    @property
    def alpha(self):
        return deref(self.alg).getFlowSmoothness()

    @property
    def gamma(self):
        return deref(self.alg).getGradientConstancyImportance()

    def __repr__(self):
        return (self.__class__.__name__ + "("
            "alpha={alpha}, "
            "gamma={gamma}, "
            "scale_factor={scale_factor}, "
            "inner_iterations={inner_iterations}, "
            "outer_iterations={outer_iterations}, "
            "solver_iterations={solver_iterations}"
            ")"
        ).format(
                alpha=self.alpha,
                gamma=self.gamma,
                scale_factor=round(self.scale_factor, 5),
                inner_iterations=self.inner_iterations,
                outer_iterations=self.outer_iterations,
                solver_iterations=self.solver_iterations
        )


cdef class CudaPyramidalLucasKanade:
    cdef Ptr[DensePyrLKOpticalFlow] alg
    cdef c_GpuMat reference_gpu, target_gpu, flow_gpu
    cdef c_Mat reference, target, flow

    def __cinit__(self,
                  int window_size = 13,
                  int max_scales = 3,
                  int iterations=30,
                  bool use_initial_flow = False
                  ):
        self.alg = DensePyrLKOpticalFlow.create(c_Size(window_size, window_size),
                                                max_scales,
                                                iterations, False)

    def __call__(self, Mat reference, Mat target) -> Mat:
        cvtColor(<InputArray>reference.c_mat,
                 <OutputArray>self.reference,
                 ColorConversionCodes.COLOR_BGR2GRAY)
        cvtColor(<InputArray>target.c_mat,
                 <OutputArray>self.target,
                 ColorConversionCodes.COLOR_BGR2GRAY)
        if self.flow_gpu.empty():
            self.flow_gpu = c_GpuMat(self.reference.rows,
                                     self.reference.cols,
                                     CV_32FC2)
        with nogil:
            self.target_gpu.upload(<InputArray> self.target)
            self.reference_gpu.upload(<InputArray> self.reference)
            deref(self.alg).calc(<InputArray>self.reference_gpu,
                                 <InputArray>self.target_gpu,
                                 <InputOutputArray>self.flow_gpu)
        flow = Mat()
        self.flow_gpu.download(<OutputArray> flow.c_mat)
        return flow

    @property
    def window_size(self):
        return deref(self.alg).getWinSize().height

    @property
    def max_scales(self):
        return deref(self.alg).getMaxLevel()

    @property
    def iterations(self):
        return deref(self.alg).getNumIters()


cdef class CudaFarnebackOpticalFlow:
    cdef Ptr[FarnebackOpticalFlow] alg
    cdef c_GpuMat reference_gpu, target_gpu, flow_gpu
    cdef c_Mat reference, target, flow, reference_float, target_float

    def __cinit__(self,
                  int scale_count = 5,
                  double scale_factor = 0.5,
                  bool fast_pyramids = False,
                  int window_size = 13,
                  int iterations = 10,
                  int neighborhood_size = 5,
                  double poly_sigma = 1.1,
                  ):
        self.alg = FarnebackOpticalFlow.create(scale_count, scale_factor,
                                               fast_pyramids, window_size,
                                               iterations, neighborhood_size, poly_sigma)

    def __call__(self, Mat reference, Mat target) -> Mat:
        cvtColor(<InputArray>reference.c_mat,
                 <OutputArray>self.reference,
                 ColorConversionCodes.COLOR_BGR2GRAY)
        cvtColor(<InputArray>target.c_mat,
                 <OutputArray>self.target,
                 ColorConversionCodes.COLOR_BGR2GRAY)
        if self.flow_gpu.empty():
            self.flow_gpu = c_GpuMat(self.reference.rows,
                                     self.reference.cols,
                                     CV_32FC2)
        with nogil:
            self.target_gpu.upload(<InputArray> self.target)
            self.reference_gpu.upload(<InputArray> self.reference)
            deref(self.alg).calc(<InputArray>self.reference_gpu,
                                 <InputArray>self.target_gpu,
                                 <InputOutputArray>self.flow_gpu)
        flow = Mat()
        self.flow_gpu.download(<OutputArray> flow.c_mat)
        return flow

    @property
    def scale_count(self) -> int:
        return deref(self.alg).getNumLevels()

    @property
    def scale_factor(self) -> float:
        return deref(self.alg).getPyrScale()

    @property
    def use_fast_pyramids(self) -> bool:
        return deref(self.alg).getFastPyramids()

    @property
    def iterations(self) -> int:
        return deref(self.alg).getNumIters()

    @property
    def neighborhood_size(self) -> int:
        return deref(self.alg).getPolyN()

    @property
    def poly_sigma(self) -> float:
        return deref(self.alg).getPolySigma()

    @property
    def window_size(self) -> int:
        return deref(self.alg).getWinSize()
