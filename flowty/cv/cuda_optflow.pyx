# cython: language_level = 3

from libcpp cimport bool
from cython.operator cimport dereference as deref
from .c_cuda_optflow cimport OpticalFlowDual_TVL1
from .core import Mat
from .core cimport Mat
from .c_core cimport Mat as c_Mat
from .c_core cimport InputArray, OutputArray, InputOutputArray, cvtColor, ColorConversionCodes, Ptr, CV_32FC2
from .c_cuda cimport GpuMat as c_GpuMat


cdef class CudaTvL1OpticalFlow:
    cdef Ptr[OpticalFlowDual_TVL1] alg
    cdef c_GpuMat reference, target, gpu_flow
    cdef c_Mat flow

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

    def __call__(self, Mat reference, Mat target):
        cvtColor(<InputArray>reference.c_mat,
                 <OutputArray>self.reference,
                 ColorConversionCodes.COLOR_BGR2GRAY)
        cvtColor(<InputArray>target.c_mat,
                 <OutputArray>self.target,
                 ColorConversionCodes.COLOR_BGR2GRAY)
        if self.flow.empty():
             self.gpu_flow = c_GpuMat(reference.rows, reference.cols, CV_32FC2)
        deref(self.alg).calc(<InputArray>self.reference,
                             <InputArray>self.target,
                             <InputOutputArray>self.gpu_flow)
        self.gpu_flow.download(<OutputArray> self.flow)
        return Mat.from_mat(self.flow)

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
             "scale_count={scale_count}, warp_count={warp_count}, "
             "iterations={iterations}, scale_step={scale_step}, "
             "use_initial_flow={use_initial_flow}"
             ")"
             ).format(
                tau=self.tau,
                lambda_=self.lambda_,
                theta=self.theta,
                epsilon=self.epsilon,
                gamma=self.gamma,
                scale_count=self.scale_count,
                warp_count=self.warp_count,
                outer_iterations=self.outer_iterations,
                inner_iterations=self.inner_iterations,
                scale_step=self.scale_step,
                median_filtering=self.median_filtering,
                use_initial_flow=self.use_initial_flow
            ))
