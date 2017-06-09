/*======================================================================================================================

    Copyright 2011, 2012, 2013, 2014, 2015 Institut fuer Neuroinformatik, Ruhr-Universitaet Bochum, Germany
 
    This file is part of cedar.

    cedar is free software: you can redistribute it and/or modify it under
    the terms of the GNU Lesser General Public License as published by the
    Free Software Foundation, either version 3 of the License, or (at your
    option) any later version.

    cedar is distributed in the hope that it will be useful, but WITHOUT ANY
    WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public
    License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with cedar. If not, see <http://www.gnu.org/licenses/>.

========================================================================================================================

    Institute:   Ruhr-Universitaet Bochum
                 Institut fuer Neuroinformatik

    File:        NeuralField.cpp

    Maintainer:  Oliver Lomp,
                 Mathis Richter,
                 Stephan Zibner
    Email:       oliver.lomp@ini.ruhr-uni-bochum.de,
                 mathis.richter@ini.ruhr-uni-bochum.de,
                 stephan.zibner@ini.ruhr-uni-bochum.de
    Date:        2011 07 04

    Description:

    Credits:

======================================================================================================================*/

// CEDAR INCLUDES
#include "NeuralField_IP.h"
#include "cedar/dynamics/gui/NeuralFieldView.h"
#include "cedar/processing/steps/Sum.h"
#include "cedar/processing/ExternalData.h"
#include "cedar/processing/exceptions.h"
#include "cedar/processing/DeclarationRegistry.h"
#include "cedar/processing/ElementDeclaration.h"
#include "cedar/auxiliaries/annotation/DiscreteMetric.h"
#include "cedar/auxiliaries/annotation/ValueRangeHint.h"
#include "cedar/auxiliaries/convolution/Convolution.h"
#include "cedar/auxiliaries/convolution/FFTW.h"
#include "cedar/auxiliaries/convolution/OpenCV.h"
#include "cedar/auxiliaries/MatData.h"
#include "cedar/auxiliaries/math/Sigmoid.h"
#include "cedar/auxiliaries/math/transferFunctions/ExpSigmoid.h"
#include "cedar/auxiliaries/kernel/Gauss.h"
#include "cedar/auxiliaries/assert.h"
#include "cedar/auxiliaries/math/tools.h"
#include "cedar/auxiliaries/Log.h"
#include "cedar/units/Time.h"
#include "cedar/units/prefixes.h"

// SYSTEM INCLUDES
#include <iostream>
#include <boost/lexical_cast.hpp>
#include <boost/make_shared.hpp>
#include <boost/units/cmath.hpp>
#include <boost/signals2/connection.hpp>
#include <QApplication>
#include <vector>
#include <set>
#include <string>

//----------------------------------------------------------------------------------------------------------------------
// internal class: icon view for DNFs
//----------------------------------------------------------------------------------------------------------------------


//----------------------------------------------------------------------------------------------------------------------
// constructors and destructor
//----------------------------------------------------------------------------------------------------------------------
NeuralField_IP::NeuralField_IP()
:
mActivation(new cedar::aux::MatData(cv::Mat::zeros(50, 50, CV_32F))),
mActivationIP(new cedar::aux::MatData(cv::Mat::zeros(50, 50, CV_32F))),
mGainIP(new cedar::aux::MatData(cv::Mat::ones(1, 1, CV_32F))),
mBiasIP(new cedar::aux::MatData(cv::Mat::zeros(1, 1, CV_32F))),
mTensorIP(new cedar::aux::MatData(cv::Mat::eye(2, 2, CV_64F))),
mSigmoidalActivation(new cedar::aux::MatData(cv::Mat::zeros(50, 50, CV_32F))),
mLateralInteraction(new cedar::aux::MatData(cv::Mat::zeros(50, 50, CV_32F))),
mInputSum(new cedar::aux::MatData(cv::Mat::zeros(50, 50, CV_32F))),
mInputNoise(new cedar::aux::MatData(cv::Mat::zeros(50, 50, CV_32F))),
mNeuralNoise(new cedar::aux::MatData(cv::Mat::zeros(50, 50, CV_32F))),
mRestingLevel
(
  new cedar::aux::DoubleParameter
  (
    this,
    "resting level",
    -5.0,
    cedar::aux::DoubleParameter::LimitType::negativeZero()
  )
),
mTau
(
  new cedar::aux::DoubleParameter
  (
    this,
    "time scale",
    100.0,
    cedar::aux::DoubleParameter::LimitType::positive()
  )
),
mGlobalInhibition
(
  new cedar::aux::DoubleParameter
  (
    this,
    "global inhibition",
    -0.01,
    cedar::aux::DoubleParameter::LimitType::negativeZero()
  )
),
// parameters
_mOutputActivation(new cedar::aux::BoolParameter(this, "activation as output", false)),
_mDiscreteMetric(new cedar::aux::BoolParameter(this, "discrete metric (workaround)", false)),
_mDimensionality
(
  new cedar::aux::UIntParameter
  (
    this,
    "dimensionality",
    2,
    cedar::aux::UIntParameter::LimitType::positiveZero(4)
  )
),
_mSizes
(
  new cedar::aux::UIntVectorParameter
  (
    this,
    "sizes",
    2,
    50,
    cedar::aux::UIntParameter::LimitType::positive(5000)
  )
),
_mInputNoiseGain
(
  new cedar::aux::DoubleParameter
  (
    this,
    "input noise gain",
    0.1,
    cedar::aux::DoubleParameter::LimitType::positiveZero()
  )
),
_mSigmoid
(
  new NeuralField_IP::SigmoidParameter
  (
    this,
    "sigmoid",
    cedar::aux::math::SigmoidPtr(new cedar::aux::math::ExpSigmoid(0.0, 1.0))
  )
),
mLearnRateIP
(
  new cedar::aux::DoubleParameter
  (
    this,
    "IP learn rate",
    0.001,
    cedar::aux::DoubleParameter::LimitType::positive()
  )
),
mMuIP
(
  new cedar::aux::DoubleParameter
  (
    this,
    "IP mean",
    0.2,
    cedar::aux::DoubleParameter::LimitType::positive()
  )
),
_mLateralKernelConvolution(new cedar::aux::conv::Convolution()),
_mNoiseCorrelationKernelConvolution(new cedar::aux::conv::Convolution())
{
  this->setAutoLockInputsAndOutputs(false);

  this->declareBuffer("activation", mActivation);
  this->declareBuffer("activation_ip", mActivationIP);
  this->declareBuffer("lateral interaction", mLateralInteraction);
  this->declareBuffer("lateral kernel", this->_mLateralKernelConvolution->getCombinedKernel());
  this->declareBuffer("neural noise kernel", this->_mNoiseCorrelationKernelConvolution->getCombinedKernel());
  this->declareBuffer("input sum", mInputSum);
  this->declareBuffer("noise", this->mInputNoise);

  this->declareBuffer("gain_ip", this->mGainIP);
  this->declareBuffer("bias_ip", this->mBiasIP);


  this->declareOutput("sigmoided activation", mSigmoidalActivation);
  this->mSigmoidalActivation->setAnnotation(cedar::aux::annotation::AnnotationPtr(new cedar::aux::annotation::ValueRangeHint(0, 1)));

  this->declareInputCollection("input");

  this->_mOutputActivation->markAdvanced();
  this->_mDiscreteMetric->markAdvanced();

  // setup default kernels
  std::vector<cedar::aux::kernel::KernelPtr> kernel_defaults;
  for (unsigned int i = 0; i < 1; i++)
  {
    cedar::aux::kernel::GaussPtr kernel
      = cedar::aux::kernel::GaussPtr(new cedar::aux::kernel::Gauss(this->getDimensionality()));
    kernel_defaults.push_back(kernel);
  }
  _mKernels = KernelListParameterPtr
              (
                new KernelListParameter
                (
                  this,
                  "lateral kernels",
                  kernel_defaults
                )
              );

  // setup noise correlation kernel
  mNoiseCorrelationKernel
    = cedar::aux::kernel::GaussPtr
      (
        new cedar::aux::kernel::Gauss
        (
          this->getDimensionality(),
          0.0 // default amplitude
        )
      );
  std::set<cedar::aux::conv::Mode::Id> allowed_convolution_modes;
  allowed_convolution_modes.insert(cedar::aux::conv::Mode::Same);

  this->addConfigurableChild("noise correlation kernel", mNoiseCorrelationKernel);
  mNoiseCorrelationKernel->markAdvanced();
  this->_mNoiseCorrelationKernelConvolution->getKernelList()->append(mNoiseCorrelationKernel);
  this->_mNoiseCorrelationKernelConvolution->setMode(cedar::aux::conv::Mode::Same);
  this->_mNoiseCorrelationKernelConvolution->setBorderType(cedar::aux::conv::BorderType::Zero);

  this->addConfigurableChild("lateral kernel convolution", _mLateralKernelConvolution);
  this->_mLateralKernelConvolution->setAllowedModes(allowed_convolution_modes);

  QObject::connect(_mSizes.get(), SIGNAL(valueChanged()), this, SLOT(dimensionSizeChanged()));
  QObject::connect(_mDimensionality.get(), SIGNAL(valueChanged()), this, SLOT(dimensionalityChanged()));
  QObject::connect(_mOutputActivation.get(), SIGNAL(valueChanged()), this, SLOT(activationAsOutputChanged()));
  QObject::connect(_mDiscreteMetric.get(), SIGNAL(valueChanged()), this, SLOT(discreteMetricChanged()));

  mKernelAddedConnection
    = this->_mKernels->connectToObjectAddedSignal(boost::bind(&NeuralField_IP::slotKernelAdded, this, _1));
  mKernelRemovedConnection
    = this->_mKernels->connectToObjectRemovedSignal
      (
        boost::bind(&NeuralField_IP::removeKernelFromConvolution, this, _1)
      );

  this->transferKernelsToConvolution();


  this->_mSigmoid->setConstant(true); // no change if the sigmoid and its parameters allowed!
  this->mBiasIP->getData() = mRestingLevel->getValue();
  this->IP_regul = 0.0001 * cv::Mat::eye(2, 2, CV_64F);
  this->IP_theta = cv::Mat(2,1,CV_64F); // vector of (delta_a; delta_b)


  // now check the dimensionality and sizes of all matrices
  this->updateMatrices();
}

//----------------------------------------------------------------------------------------------------------------------
// methods
//----------------------------------------------------------------------------------------------------------------------

void NeuralField_IP::discreteMetricChanged()
{
  std::vector<cedar::aux::DataPtr> data_items;
  data_items.push_back(this->mActivation);
  data_items.push_back(this->mSigmoidalActivation);
  data_items.push_back(this->mLateralInteraction);

  for (size_t i = 0; i < data_items.size(); ++i)
  {
    if (this->_mDiscreteMetric->getValue() == true)
    {
      data_items.at(i)->setAnnotation(boost::make_shared<cedar::aux::annotation::DiscreteMetric>());
    }
    else
    {
      data_items.at(i)->removeAnnotations<cedar::aux::annotation::DiscreteMetric>();
    }
  }
}

bool NeuralField_IP::activationIsOutput() const
{
  return this->_mOutputActivation->getValue();
}

void NeuralField_IP::activationAsOutputChanged()
{
  bool act_is_output = this->activationIsOutput();
  static std::string slot_name = "activation";

  if (act_is_output)
  {
    if (this->hasBufferSlot(slot_name))
    {
      this->removeBufferSlot(slot_name);
    }

    if (!this->hasOutputSlot(slot_name))
    {
      this->declareOutput(slot_name, this->mActivation);
    }
  }
  else
  {
    if (this->hasOutputSlot(slot_name))
    {
      this->removeOutputSlot(slot_name);
    }

    if (!this->hasBufferSlot(slot_name))
    {
      this->declareBuffer(slot_name, this->mActivation);
    }
  }
}

void NeuralField_IP::slotKernelAdded(size_t kernelIndex)
{
  cedar::aux::kernel::KernelPtr kernel = this->_mKernels->at(kernelIndex);
  this->addKernelToConvolution(kernel);
}

void NeuralField_IP::transferKernelsToConvolution()
{
  this->getConvolution()->getKernelList()->clear();
  for (size_t kernel = 0; kernel < this->_mKernels->size(); ++ kernel)
  {
    this->addKernelToConvolution(this->_mKernels->at(kernel));
  }
}

void NeuralField_IP::addKernelToConvolution(cedar::aux::kernel::KernelPtr kernel)
{
  kernel->setDimensionality(this->getDimensionality());
  this->getConvolution()->getKernelList()->append(kernel);
}

void NeuralField_IP::removeKernelFromConvolution(size_t index)
{
  this->getConvolution()->getKernelList()->remove(index);
}

void NeuralField_IP::readConfiguration(const cedar::aux::ConfigurationNode& node)
{
  // disconnect kernel slots (kernels first have to be loaded completely)
  mKernelAddedConnection.disconnect();
  mKernelRemovedConnection.disconnect();

  this->cedar::proc::Step::readConfiguration(node);

  this->transferKernelsToConvolution();

  // reconnect slots
  mKernelAddedConnection
    = this->_mKernels->connectToObjectAddedSignal(boost::bind(&NeuralField_IP::slotKernelAdded, this, _1));
  mKernelRemovedConnection
    = this->_mKernels->connectToObjectRemovedSignal
      (
        boost::bind(&NeuralField_IP::removeKernelFromConvolution, this, _1)
      );

  // legacy code for reading kernels with the old format
  cedar::aux::ConfigurationNode::const_assoc_iterator iter = node.find("numberOfKernels");
  if (iter != node.not_found())
  {
    unsigned int num_kernels = iter->second.get_value<unsigned int>();

    if (num_kernels > 0)
    {
      cedar::aux::LogSingleton::getInstance()->warning
      (
        "Reading kernels for field \"" + this->getName() + "\" with legacy mode. "
        "This overrides all kernels previously set!",
        "cedar::dyn::NeuralField::readConfiguration(const cedar::aux::ConfigurationNode&)"
      );

      /* we have to clear everything here because it is not known whether the kernels already in the list are default
         values or values read from the configuration.
       */
      this->_mKernels->clear();
    }

    for (unsigned int i = 0; i < num_kernels; ++i)
    {
      // find the configuration node for the kernel
      cedar::aux::ConfigurationNode::const_assoc_iterator kernel_iter;
      kernel_iter = node.find("lateralKernel" + cedar::aux::toString(i));

      // check if the kernel node was found
      if (kernel_iter != node.not_found())
      {
        // the old kernels were all Gauss kernels
        cedar::aux::kernel::GaussPtr kernel (new cedar::aux::kernel::Gauss());

        // read the kernel's configuration
        kernel->readConfiguration(kernel_iter->second);

        // add the kernel to the managed list
        this->_mKernels->pushBack(kernel);
      }
      else
      {
        cedar::aux::LogSingleton::getInstance()->warning
        (
          "Could not find legacy kernel description for kernel " + cedar::aux::toString(i)
           + " in field \"" + this->getName() + "\". Skipping kernel!",
          "cedar::dyn::NeuralField::readConfiguration(const cedar::aux::ConfigurationNode&)"
        );
      }
    }
  }

  // legacy reading of the sigmoid
  cedar::aux::ConfigurationNode::const_assoc_iterator sigmoid_iter = node.find("sigmoid");
  if (sigmoid_iter != node.not_found())
  {
    cedar::aux::ConfigurationNode::const_assoc_iterator type_iter = sigmoid_iter->second.find("type");

    // if there is no type entry in the sigmoid, this must be the old format
    if (type_iter == sigmoid_iter->second.not_found())
    {
      /* If we get to this point, the sigmoid should already contain a pointer to a proper object, but with the default
         settings. Thus, we let it read the values from the sigmoid node.
       */
      CEDAR_DEBUG_ASSERT(this->_mSigmoid->getValue());

      this->_mSigmoid->getValue()->readConfiguration(sigmoid_iter->second);
    }
  }
}


void NeuralField_IP::reset()
{
  // these buffers are still locked automatically
  this->mActivation->getData() = mRestingLevel->getValue();
  this->mActivationIP->getData() = mRestingLevel->getValue();
  this->mLateralInteraction->getData() = cv::Scalar(0);
  this->mInputNoise->getData() = cv::Scalar(0);
  this->mNeuralNoise->getData() = cv::Scalar(0);

  this->lockOutputs();
  this->mSigmoidalActivation->getData() = cv::Scalar(0);
  this->unlockOutputs();
  
  this->mGainIP->getData() = cv::Scalar(1.0);
  this->mBiasIP->getData() = this->mRestingLevel->getValue();
  this->mTensorIP->getData() = cv::Mat::eye(2,2,CV_64F);
  
}


cedar::proc::DataSlot::VALIDITY NeuralField_IP::determineInputValidity
                                                         (
                                                           cedar::proc::ConstDataSlotPtr slot,
                                                           cedar::aux::ConstDataPtr data
                                                         ) const
{
  if (slot->getRole() == cedar::proc::DataRole::INPUT && slot->getName() == "input")
  {
    if (cedar::aux::ConstMatDataPtr input = boost::dynamic_pointer_cast<const cedar::aux::MatData>(data))
    {
      if (!this->isMatrixCompatibleInput(input->getData()))
      {
        return cedar::proc::DataSlot::VALIDITY_ERROR;
      }
      else
      {
        return cedar::proc::DataSlot::VALIDITY_VALID;
      }
    }
  }

  return cedar::proc::DataSlot::VALIDITY_ERROR;
}


void NeuralField_IP::eulerStep(const cedar::unit::Time& time)
{
  // get all members needed for the Euler step
  cv::Mat& lateral_interaction = this->mLateralInteraction->getData();
  cv::Mat& input_noise = this->mInputNoise->getData();
  cv::Mat& neural_noise = this->mNeuralNoise->getData();
  cv::Mat& u = this->mActivation->getData();
  cv::Mat& uIP = this->mActivationIP->getData();
  cv::Mat& gainIP = this->mGainIP->getData();
  cv::Mat& biasIP = this->mBiasIP->getData();
  cv::Mat& tensorIP = this->mTensorIP->getData();
  cv::Mat& input_sum = this->mInputSum->getData();
  const double& eta = this->mLearnRateIP->getValue();
  const double& mu = this->mMuIP->getValue();
  const double& h = mRestingLevel->getValue();
  const double& tau = mTau->getValue();
  const double& global_inhibition = mGlobalInhibition->getValue();
  
  double delta, activation, input;
  int pos_max[2];
  double timeConst = time / cedar::unit::Time(tau * cedar::unit::milli * cedar::unit::seconds);
  double tensor_decay = timeConst * 1.0/1000.0;

  
  boost::shared_ptr<QReadLocker> activation_read_locker;
  if (this->activationIsOutput())
  {
    activation_read_locker = boost::shared_ptr<QReadLocker>(new QReadLocker(&this->mActivation->getLock()));
  }

  QWriteLocker sigmoid_u_lock(&this->mSigmoidalActivation->getLock());
  cv::Mat& sigmoid_u = this->mSigmoidalActivation->getData();
  
    // if the neural noise correlation kernel has an amplitude != 0, create new random values and convolve
  if (mNoiseCorrelationKernel->getAmplitude() != 0.0)
  {
    cv::randn(neural_noise, cv::Scalar(0), cv::Scalar(1));
    neural_noise = this->_mNoiseCorrelationKernelConvolution->convolve(neural_noise);

    //!@todo document why this has to use sqrt(time) for noise
    sigmoid_u = _mSigmoid->getValue()->compute
                (
                  uIP
                  + sqrt(time / (1.0 * cedar::unit::second)) * neural_noise
                );
  }
  else
  {
    // calculate output
    sigmoid_u = _mSigmoid->getValue()->compute(uIP);
  }
  sigmoid_u_lock.unlock();

  QReadLocker sigmoid_u_readlock(&this->mSigmoidalActivation->getLock());
  lateral_interaction = this->_mLateralKernelConvolution->convolve(sigmoid_u);

  this->updateInputSum();

  CEDAR_ASSERT(u.size == sigmoid_u.size);
  CEDAR_ASSERT(u.size == lateral_interaction.size);
  CEDAR_ASSERT(u.size == input_sum.size);
  
  ///-------------------------------------------------------------------
  /// adapt gain and bias via IP
  // determine field input and output: max output -> according input
  cv::minMaxIdx(sigmoid_u, NULL, &activation, NULL, &pos_max[0]);
  //input = input_sum.at<float>(pos_max[0],pos_max[1]);
  input = u.at<float>(pos_max[0],pos_max[1]);
  
  //std::cerr<< "NeuralField_IP in: "<<input<<"; out: "<< activation<<";"<<std::endl;
  
  // determine changes in gain and bias (gradient descent)
  delta = 1 - (2 + (1.0 / mu) )*activation + (1.0 / mu)*pow(activation,2);  
  this->IP_theta.at<double>(1,0) = delta;
  this->IP_theta.at<double>(0,0) = ( 1.0 / gainIP.at<float>(0,0) + input * delta );
  
  // estimate the Riemannian metric tensor (Fisher information) for natrual gradient descent
  /// TODO: switch to the online estimate of the tensor-inverse (see park et. al. 1999)
  tensorIP = (1 - tensor_decay) * tensorIP + tensor_decay * (this->IP_theta * this->IP_theta.t());
  // transform gradient vector into Riemannian space
  this->IP_theta_F = (tensorIP + IP_regul).inv(cv::DECOMP_CHOLESKY) * this->IP_theta;
  
  //std::cerr<< "NeuralField_IP da: "<< this->IP_theta_F.at<double>(0,0) <<"; db: "<< this->IP_theta_F.at<double>(1,0) <<"; d: "<<delta<<"; "<<std::endl;
  
  //*
  // NG:
  gainIP.at<float>(0,0) += timeConst * eta * this->IP_theta_F.at<double>(0,0);
  biasIP.at<float>(0,0) += timeConst * eta * this->IP_theta_F.at<double>(1,0);
  
  /*/
  // no NG
  gainIP.at<float>(0,0) += timeConst * eta * this->IP_theta.at<double>(0,0);
  biasIP.at<float>(0,0) += timeConst * eta * this->IP_theta.at<double>(1,0);
  //*/
  
  /// --- IP end
  ///-------------------------------------------------------------------

  // the field equation
  //cv::Mat d_u = -u + h + lateral_interaction + global_inhibition * cv::sum(sigmoid_u)[0] + input_sum;
  cv::Mat d_u = -u + lateral_interaction + global_inhibition * cv::sum(sigmoid_u)[0] + input_sum;

  boost::shared_ptr<QWriteLocker> activation_write_locker;
  if (this->activationIsOutput())
  {
    activation_read_locker->unlock();
    activation_write_locker = boost::shared_ptr<QWriteLocker>(new QWriteLocker(&this->mActivation->getLock()));
  }

  cv::randn(input_noise, cv::Scalar(0), cv::Scalar(1));

  // integrate one time step
  u += timeConst * d_u
       + (sqrt(time / (cedar::unit::Time(1.0 * cedar::unit::milli * cedar::unit::seconds))) / tau)
           * _mInputNoiseGain->getValue() * input_noise;
  uIP = gainIP.at<float>(0,0) * u + biasIP.at<float>(0,0);
}

void NeuralField_IP::updateInputSum()
{
  cedar::proc::steps::Sum::sumSlot(this->getInputSlot("input"), this->mInputSum->getData(), true);
}

bool NeuralField_IP::isMatrixCompatibleInput(const cv::Mat& matrix) const
{
  if (matrix.type() != CV_32F)
  {
    return false;
  }

  unsigned int matrix_dim = cedar::aux::math::getDimensionalityOf(matrix);
  return matrix_dim == 0
         ||
         (
           this->getDimensionality() == matrix_dim
           && cedar::aux::math::matrixSizesEqual(matrix, this->getFieldActivation()->getData())
         );
}

void NeuralField_IP::dimensionalityChanged()
{
  this->_mSizes->resize(this->getDimensionality(), _mSizes->getDefaultValue());
#ifdef CEDAR_USE_FFTW
  if (this->getDimensionality() >= 3)
  {
    this->_mLateralKernelConvolution->setEngine(cedar::aux::conv::FFTWPtr(new cedar::aux::conv::FFTW()));
    this->_mNoiseCorrelationKernelConvolution->setEngine(cedar::aux::conv::FFTWPtr(new cedar::aux::conv::FFTW()));
  }
#endif // CEDAR_USE_FFTW
  this->updateMatrices();
}

void NeuralField_IP::dimensionSizeChanged()
{
  this->updateMatrices();
}

void NeuralField_IP::updateMatrices()
{
  int dimensionality = static_cast<int>(this->getDimensionality());

  std::vector<int> sizes(dimensionality);
  for (int dim = 0; dim < dimensionality; ++dim)
  {
    sizes[dim] = _mSizes->at(dim);
  }
  // check if matrices become too large
  double max_size = 1.0;
  for (int dim = 0; dim < dimensionality; ++dim)
  {
    max_size *= sizes[dim];
    if (max_size > std::numeric_limits<unsigned int>::max()/500.0) // heuristics
    {
      CEDAR_THROW(cedar::aux::RangeException, "cannot handle matrices of this size");
      return;
    }
  }
  this->lockAll();
  if (dimensionality == 0)
  {
    this->mActivation->getData() = cv::Mat(1, 1, CV_32F, cv::Scalar(mRestingLevel->getValue()));
    this->mActivationIP->getData() = cv::Mat(1, 1, CV_32F, cv::Scalar(mRestingLevel->getValue()));
    this->mSigmoidalActivation->getData() = cv::Mat(1, 1, CV_32F, cv::Scalar(0));
    this->mLateralInteraction->getData() = cv::Mat(1, 1, CV_32F, cv::Scalar(0));
    this->mInputNoise->getData() = cv::Mat(1, 1, CV_32F, cv::Scalar(0));
    this->mNeuralNoise->getData() = cv::Mat(1, 1, CV_32F, cv::Scalar(0));
    this->mInputSum->setData(cv::Mat(1, 1, CV_32F, cv::Scalar(0)));
  }
  else if (dimensionality == 1)
  {
    this->mActivation->getData() = cv::Mat(sizes[0], 1, CV_32F, cv::Scalar(mRestingLevel->getValue()));
    this->mActivationIP->getData() = cv::Mat(sizes[0], 1, CV_32F, cv::Scalar(mRestingLevel->getValue()));
    this->mSigmoidalActivation->getData() = cv::Mat(sizes[0], 1, CV_32F, cv::Scalar(0));
    this->mLateralInteraction->getData() = cv::Mat(sizes[0], 1, CV_32F, cv::Scalar(0));
    this->mInputNoise->getData() = cv::Mat(sizes[0], 1, CV_32F, cv::Scalar(0));
    this->mNeuralNoise->getData() = cv::Mat(sizes[0], 1, CV_32F, cv::Scalar(0));
    this->mInputSum->setData(cv::Mat(sizes[0], 1, CV_32F, cv::Scalar(0)));
  }
  else
  {
    this->mActivation->getData() = cv::Mat(dimensionality,&sizes.at(0), CV_32F, cv::Scalar(mRestingLevel->getValue()));
    this->mActivationIP->getData() = cv::Mat(dimensionality,&sizes.at(0), CV_32F, cv::Scalar(mRestingLevel->getValue()));
    this->mSigmoidalActivation->getData() = cv::Mat(dimensionality, &sizes.at(0), CV_32F, cv::Scalar(0));
    this->mLateralInteraction->getData() = cv::Mat(dimensionality, &sizes.at(0), CV_32F, cv::Scalar(0));
    this->mInputNoise->getData() = cv::Mat(dimensionality, &sizes.at(0), CV_32F, cv::Scalar(0));
    this->mNeuralNoise->getData() = cv::Mat(dimensionality, &sizes.at(0), CV_32F, cv::Scalar(0));
    this->mInputSum->setData(cv::Mat(dimensionality, &sizes.at(0), CV_32F, cv::Scalar(0)));
  }
  this->unlockAll();
  if (dimensionality > 0) // only adapt kernel in non-0D case
  {
    for (unsigned int i = 0; i < _mKernels->size(); i++)
    {
      this->_mKernels->at(i)->setDimensionality(dimensionality);
    }
    this->mNoiseCorrelationKernel->setDimensionality(dimensionality);
  }

  this->revalidateInputSlot("input");

  if (this->activationIsOutput())
  {
    this->emitOutputPropertiesChangedSignal("activation");
  }
  this->emitOutputPropertiesChangedSignal("sigmoided activation");
}

void NeuralField_IP::onStart()
{
  this->_mDimensionality->setConstant(true);
  this->_mSizes->setConstant(true);
}

void NeuralField_IP::onStop()
{
  this->_mDimensionality->setConstant(false);
  this->_mSizes->setConstant(false);
}
