# cython: language_level=3

from cython.operator cimport dereference as deref
from libcpp cimport bool
from libcpp.memory cimport shared_ptr
from ..cv.c_core cimport Mat as c_Mat, InputArray, OutputArray, \
    InputOutputArray, CV_32FC2
from ..cv.c_imgproc cimport cvtColor, ColorConversionCodes
from ..cv.core cimport Mat
from ..cv.c_optflow cimport DualTVL1OpticalFlow as c_DualTVL1OpticalFlow

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
    cdef shared_ptr[c_DualTVL1OpticalFlow] alg
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
                  int outer_iterations=10,
                  int inner_iterations=30,
                  int median_filtering=5,
                  bool use_initial_flow=False):
        self.alg = c_DualTVL1OpticalFlow.create(tau, lambda_, theta, scale_count, warp_count, epsilon,
            inner_iterations, outer_iterations, scale_step, gamma, median_filtering, use_initial_flow)

    def __call__(self, Mat reference, Mat target):
        cvtColor(<InputArray>reference.c_mat,
                 <OutputArray>self.reference,
                 ColorConversionCodes.COLOR_BGR2GRAY)
        cvtColor(<InputArray>target.c_mat,
                 <OutputArray>self.target,
                 ColorConversionCodes.COLOR_BGR2GRAY)
        if self.flow.empty():
             self.flow = c_Mat(reference.rows, reference.cols, CV_32FC2)
        deref(self.alg).calc(<InputArray>self.reference, <InputArray>self.target, <InputOutputArray>self.flow)
        return Mat.from_mat(self.flow.clone())


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
