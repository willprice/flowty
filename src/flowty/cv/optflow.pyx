# cython: language_level=3
from pathlib import Path
from typing import Union

from cython.operator cimport dereference as deref
import numpy as np
cimport numpy as np
from libcpp cimport bool
from libcpp.string cimport string
from .c_core cimport Ptr, String, Mat as c_Mat, InputArray, OutputArray, \
    InputOutputArray, CV_32FC2, CV_32FC1
from .c_imgproc cimport cvtColor, ColorConversionCodes
from .core cimport Mat
from .c_optflow cimport DenseOpticalFlow as c_DenseOpticalFlow, \
     DualTVL1OpticalFlow as c_DualTVL1OpticalFlow, \
     FarnebackOpticalFlow as c_FarnebackOpticalFlow, \
     DISOpticalFlow as c_DISOpticalFlow, PRESET_ULTRAFAST, PRESET_FAST, PRESET_MEDIUM, \
     VariationalRefinement as c_VariationalRefinement, \
     writeOpticalFlow, readOpticalFlow


cdef compute_flow(Ptr[c_DenseOpticalFlow] algorithm,
                  Mat reference_rgb,
                  Mat target_rgb,
                  c_Mat& reference_gray,
                  c_Mat& target_gray,
                  c_Mat& flow):
    cvtColor(<InputArray> reference_rgb.c_mat, <OutputArray> reference_gray,
             ColorConversionCodes.COLOR_BGR2GRAY)
    cvtColor(<InputArray> target_rgb.c_mat, <OutputArray> target_gray,
             ColorConversionCodes.COLOR_BGR2GRAY)
    deref(algorithm).calc(<InputArray> reference_gray, <InputArray> target_gray,
                          <InputOutputArray> flow)

cdef class TvL1OpticalFlow:
    """
    Args:
        tau:  Time step of the numerical scheme.
        lambda_:  Weight parameter for the data term, attachment parameter.
            This is the most relevant parameter, which determines the smoothness of
            the output. The smaller this parameter is, the smoother the solutions
            we obtain. It depends on the range of motions of the images, so its
            value should be adapted to each image sequence.
        theta: Weight parameter for (u - v)\^2, tightness parameter. It serves
            as a link between the attachment and the regularization terms. In
            theory, it should have a small value in order to maintain both parts in
            correspondence. The method is stable for a large range of values of
            this parameter.
        epsilon: Stopping criterion threshold used in the numerical scheme,
             which is a trade-off between precision and running time. A small value will
             yield more accurate solutions at the expense of a slower convergence.
        scale_count: Number of scales used to create the pyramid of images.
        warp_count: Number of warpings per scale.Represents the number of
             times that I1(x+u0) and grad( I1(x+u0) ) are computed per scale. This is a
             parameter that assures the stability of the method. It also affects the running
             time, so it is a compromise between speed and accuracy.
    """
    cdef Ptr[c_DualTVL1OpticalFlow] alg
    cdef c_Mat reference, target

    def __cinit__(self,
                  double tau=0.25,
                  double lambda_=0.15,
                  double theta=0.3,
                  double epsilon=0.01,
                  double gamma=0.0,
                  double scale_step=0.8,
                  int scale_count=5,
                  int warp_count=5,
                  int outer_iterations=10,
                  int inner_iterations=30,
                  int median_filtering=5,
                  bool use_initial_flow=False):
        self.alg = c_DualTVL1OpticalFlow.create(tau, lambda_, theta, scale_count, warp_count, epsilon,
            inner_iterations, outer_iterations, scale_step, gamma, median_filtering, use_initial_flow)

    def __call__(self, Mat reference, Mat target):
        flow = Mat()
        compute_flow(<Ptr[c_DenseOpticalFlow]>self.alg, reference, target,
                     self.reference,
                     self.target,
                     flow.c_mat)
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
        return deref(self.alg).getScalesNumber()

    @property
    def warp_count(self):
        return deref(self.alg).getWarpingsNumber()

    @property
    def outer_iterations(self):
        return deref(self.alg).getOuterIterations()

    @property
    def inner_iterations(self):
        return deref(self.alg).getInnerIterations()

    @property
    def median_filtering(self):
        return deref(self.alg).getMedianFiltering()

    @property
    def use_initial_flow(self):
        return deref(self.alg).getUseInitialFlow()

    def __repr__(self):
        return (self.__class__.__name__ +
            ("("
             "tau={tau}, lambda_={lambda_}, theta={theta}, epsilon={epsilon}, gamma={gamma}, "
             "scale_step={scale_step}, scale_count={scale_count}, warp_count={warp_count}, "
             "outer_iterations={outer_iterations}, inner_iterations={inner_iterations}, "
             "median_filtering={median_filtering}, use_initial_flow={use_initial_flow}"
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


cdef class FarnebackOpticalFlow:
    cdef Ptr[c_FarnebackOpticalFlow] alg
    cdef c_Mat reference, target, flow

    def __cinit__(self,
                  int scale_count = 5,
                  double scale_factor = 0.5,
                  bool fast_pyramids = False,
                  int window_size = 13,
                  int iterations = 10,
                  int neighborhood_size = 5,
                  double poly_sigma = 1.1,
                  ):
        self.alg = c_FarnebackOpticalFlow.create(scale_count, scale_factor,
                                                 fast_pyramids, window_size,
                                                 iterations, neighborhood_size, poly_sigma)


    def __call__(self, Mat reference, Mat target):
        flow = Mat()
        compute_flow(<Ptr[c_DenseOpticalFlow]>self.alg, reference, target,
                     self.reference,
                     self.target,
                     flow.c_mat)
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



cdef class VariationalRefinementOpticalFlow:
    cdef Ptr[c_VariationalRefinement] alg
    cdef c_Mat reference, target, flow, reference_float, target_float

    def __cinit__(self,
                  fixed_point_iterations = 5,
                  sor_iterations = 5,
                  alpha = 20.0,
                  delta = 5.0,
                  gamma = 10.0,
                  omega = 1.6,
                  ):
        self.alg = c_VariationalRefinement.create()
        deref(self.alg).setFixedPointIterations(fixed_point_iterations)
        deref(self.alg).setSorIterations(sor_iterations)
        deref(self.alg).setAlpha(alpha)
        deref(self.alg).setDelta(delta)
        deref(self.alg).setGamma(gamma)
        deref(self.alg).setOmega(omega)


    def __call__(self, Mat reference, Mat target):
        cvtColor(<InputArray>reference.c_mat,
                 <OutputArray>self.reference,
                 ColorConversionCodes.COLOR_BGR2GRAY)
        cvtColor(<InputArray>target.c_mat,
                 <OutputArray>self.target,
                 ColorConversionCodes.COLOR_BGR2GRAY)
        self.target.convertTo(<OutputArray> self.target_float, CV_32FC1, 1.0 / 255.0)
        self.reference.convertTo(<OutputArray> self.reference_float, CV_32FC1,
                                 1.0 / 255.0)
        flow = Mat(self.reference.rows, self.reference.cols, CV_32FC2)
        # Without setting the flow matrix to zero, you get weird pattern artifacts.
        flow.c_mat.setTo(<InputArray> 0)
        deref(self.alg).calc(<InputArray>self.reference_float,
                             <InputArray>self.target_float,
                             <InputOutputArray> flow.c_mat)
        return flow

    @property
    def sor_iterations(self) -> int:
        return deref(self.alg).getSorIterations()

    @property
    def fixed_point_iterations(self) -> int:
        return deref(self.alg).getFixedPointIterations()

    @property
    def alpha(self) -> float:
        return deref(self.alg).getAlpha()

    @property
    def delta(self) -> float:
        return deref(self.alg).getDelta()

    @property
    def gamma(self) -> float:
        return deref(self.alg).getGamma()

    @property
    def omega(self) -> float:
        return deref(self.alg).getOmega()

cdef class DenseInverseSearchOpticalFlow:
    cdef Ptr[c_DISOpticalFlow] alg
    cdef c_Mat reference, target
    _preset_to_enum = {
        'ultrafast': PRESET_ULTRAFAST,
        'fast': PRESET_FAST,
        'medium': PRESET_MEDIUM
    }

    def __cinit__(self, str preset = 'ultrafast',
                  finest_scale=None,
                  gradient_descent_iterations=None,
                  patch_size=None,
                  patch_stride=None,
                  use_mean_normalization=None,
                  use_spatial_propagation=None,
                  alpha=None,
                  delta=None,
                  gamma=None,
                  variational_refinement_iterations=None):
        print(preset)
        preset_int = self._preset_to_enum[preset.lower()]
        self.alg = c_DISOpticalFlow.create(preset_int)

        if finest_scale is not None:
            self.finest_scale = finest_scale
        if gradient_descent_iterations is not None:
            self.gradient_descent_iterations = gradient_descent_iterations
        if patch_size is not None:
            self.patch_size = patch_size
        if patch_stride is not None:
            self.patch_stride = patch_stride
        if use_mean_normalization is not None:
            self.use_mean_normalization = use_mean_normalization
        if use_spatial_propagation is not None:
            self.use_spatial_propagation = use_spatial_propagation
        if alpha is not None:
            self.alpha = alpha
        if delta is not None:
            self.delta = delta
        if gamma is not None:
            self.gamma = gamma
        if variational_refinement_iterations is not None:
            self.variational_refinement_iterations = variational_refinement_iterations


    def __call__(self, Mat reference, Mat target):
        flow = Mat()
        compute_flow(<Ptr[c_DenseOpticalFlow]>self.alg, reference, target,
                     self.reference,
                     self.target,
                     flow.c_mat)
        return flow

    @property
    def finest_scale(self) -> int:
        return deref(self.alg).getFinestScale()

    @finest_scale.setter
    def finest_scale(self, int finest_scale):
        deref(self.alg).setFinestScale(finest_scale)

    @property
    def gradient_descent_iterations(self) -> int:
        return deref(self.alg).getGradientDescentIterations()

    @gradient_descent_iterations.setter
    def gradient_descent_iterations(self, int iterations):
        deref(self.alg).setGradientDescentIterations(iterations)

    @property
    def patch_size(self) -> int:
        return deref(self.alg).getPatchSize()

    @patch_size.setter
    def patch_size(self, int patch_size):
        deref(self.alg).setPatchSize(patch_size)

    @property
    def patch_stride(self) -> int:
        return deref(self.alg).getPatchStride()

    @patch_stride.setter
    def patch_stride(self, int patch_stride):
        deref(self.alg).setPatchStride(patch_stride)

    @property
    def use_mean_normalization(self) -> bool:
        return deref(self.alg).getUseMeanNormalization()

    @use_mean_normalization.setter
    def use_mean_normalization(self, bool on):
        deref(self.alg).setUseMeanNormalization(on)

    @property
    def use_spatial_propagation(self) -> bool:
        return deref(self.alg).getUseSpatialPropagation()

    @use_spatial_propagation.setter
    def use_spatial_propagation(self, bool on):
        deref(self.alg).setUseSpatialPropagation(on)

    @property
    def alpha(self) -> float:
        return deref(self.alg).getVariationalRefinementAlpha()

    @alpha.setter
    def alpha(self, float alpha):
        deref(self.alg).setVariationalRefinementAlpha(alpha)

    @property
    def delta(self) -> float:
        return deref(self.alg).getVariationalRefinementDelta()

    @delta.setter
    def delta(self, float delta):
        deref(self.alg).setVariationalRefinementDelta(delta)


    @property
    def gamma(self) -> float:
        return deref(self.alg).getVariationalRefinementGamma()

    @gamma.setter
    def gamma(self, float gamma):
        deref(self.alg).setVariationalRefinementGamma(gamma)

    @property
    def variational_refinement_iterations(self) -> int:
        return deref(self.alg).getVariationalRefinementIterations()

    @variational_refinement_iterations.setter
    def variational_refinement_iterations(self, int iterations):
        deref(self.alg).setVariationalRefinementIterations(iterations)

def read_flo(str path) -> Mat:
    if isinstance(path, Path):
        path = str(path)
    cdef c_Mat flow = readOpticalFlow(String(<string> path.encode('utf8')))
    return Mat.from_mat(flow)

def write_flo(flow, path) -> None:
    if isinstance(path, Path):
        path = str(path)
    if isinstance(flow, np.ndarray):
        ok = _write_flo_np(flow, path)
    elif isinstance(flow, Mat):
        ok = _write_flo_mat(flow, path)
    if not ok:
        raise RuntimeError("Unable to write flow to {}".format(path))

def _write_flo_mat(Mat flow, str path):
    return writeOpticalFlow(String(<string> path.encode('utf8')),
                            <InputArray> flow.c_mat)

def _write_flo_np(flow, str path):
    if flow.dtype != np.float32:
        flow = flow.astype(np.float32)
    if flow.ndim != 3:
        raise ValueError("Expected 3 dimensional array but had shape: {}".format(
                flow.shape))
    if flow.shape[-1] != 2:
        raise ValueError("Expected 2 channel image, but was {} channel".format(
                flow.shape[-1]))
    cdef Mat flow_mat = Mat.fromarray(flow)
    return writeOpticalFlow(String(<string> path.encode('utf8')),
                            <InputArray> flow_mat.c_mat)