package weka.classifiers.meta;

import weka.classifiers.Classifier;
import weka.classifiers.RandomizableIteratedSingleClassifierEnhancer;
import weka.classifiers.rules.ZeroR;
import weka.core.*;

import java.io.Serializable;
import java.util.Enumeration;
import java.util.Random;

/**
 * This class provides an implementation of the basic XGBoost algorithm, excluding the base learner implementation.
 * The default base learner is XGBoostTree. The class extends RandomizableIteratedClassifierEnhancer to inherit
 * member variables and corresponding option handling for specifying the number of boosting iterations to run,
 * and the seed for the random number generator that is used to initialise the random number generator of the base
 * learner, which is assumed to also implement the Randomizable interface (e.g., by extending the class
 * RandomizableClassifier). The base learner, e.g., XGBoostTree, must also implement the WeightedInstancesHandler
 * interface because the values of the Hessian for each instance are provided to the base learner as instance
 * weights. The class values of the instances passed to the base learner are the corresponding negative gradients.
 */
public class XGBoost extends RandomizableIteratedSingleClassifierEnhancer implements AdditionalMeasureProducer {

    /**  Returns the capabilities of the classifier: numeric predictors and numeric or binary classes. */
    public Capabilities getCapabilities() {
        Capabilities result = super.getCapabilities();
        result.disableAll();
        result.enable(Capabilities.Capability.NUMERIC_ATTRIBUTES);
        result.enable(Capabilities.Capability.NUMERIC_CLASS);
        result.enable(Capabilities.Capability.BINARY_CLASS);
        return result;
    }

    /** The default number of iterations to perform (XGBoost uses a 100 iterations by default). */
    protected int defaultNumberOfIterations() {
        return 100;
    }

    /** String describing default classifier. */
    protected String defaultClassifierString() { return "weka.classifiers.trees.XGBoostTree"; }

    /** Default constructor setting the default classifier. */
    public XGBoost() { m_Classifier = new weka.classifiers.trees.XGBoostTree(); }

    /**
     * Provides an enumeration of the additional measures supplied by the base learner (if any).
     */
    public Enumeration<String> enumerateMeasures() {
        if (m_Classifier instanceof AdditionalMeasureProducer) {
            return ((AdditionalMeasureProducer) m_Classifier).enumerateMeasures();
        } else {
            return new Enumeration<String>() {
                public boolean hasMoreElements() { return false; }
                public String nextElement() { return null; }
            };
        }
    }

    /**
     * Provides the sum of the specified measure across all base models (and throws an exception if it does not exist).
     */
    public double getMeasure(String measureName) throws IllegalArgumentException {
        if (m_Classifier instanceof AdditionalMeasureProducer) {
            double sum = 0;
            for (Classifier classifier : m_Classifiers) {
                sum += ((AdditionalMeasureProducer)classifier).getMeasure(measureName);
            }
            return sum;
        } else {
            throw new IllegalArgumentException("Measure " + measureName + " not supported.");
        }
    }

    /** Interface implemented by loss functions. */
    private interface LossFunction extends Serializable {
        /** Returns the negative gradient for the given instance. */
        double negativeGradient(Instance i, double pred);
        /** Returns the Hessian for the given instance. */
        double hessian(Instance i, double pred);
        /** Returns the prediction in a suitable form. */
        double prediction(double pred);
    }

    /** Class implementing the squared error regression loss. */
    private class SquaredError implements LossFunction {
        public double negativeGradient(Instance i, double pred) { return 2 * (i.classValue() - pred); }
        public double hessian(Instance i, double pred) { return 2.0; }
        public double prediction(double pred) {
            return pred;
        }
    }

    /** Class implementing the binary log loss for classification. */
    private class LogLoss implements LossFunction {
        public double prob(double pred) {
            return 1.0 / (1.0 + Math.exp(-pred));
        }
        public double negativeGradient(Instance i, double pred) { return i.classValue() - prob(pred);}
        public double hessian(Instance i, double pred) { return Double.max(prob(pred)  * (1.0 - prob(pred)), 1e-16); }
        public double prediction(double pred) { return prob(pred); }
    }

    /** The loss function to use. */
    private LossFunction loss;
    /** Variable to hold data that can be passed to the XGBoostTree learner. */
    private Instances xgBoostData;
    /** The classifier used to initialise the ensemble (a ZeroR model) */
    private Classifier initialClassifier;

    private void buildBaseClassifier(Classifier classifier, Instances data, double[] previousPredictions,
                                     Instances xgBoostData) throws Exception {
        for (int i = 0; i < data.numInstances(); i++) {
            Instance instance = data.instance(i);
            Instance xgBoostInstance = (Instance)instance.copy();
            xgBoostInstance.setClassValue(loss.negativeGradient(instance, previousPredictions[i]));
            xgBoostInstance.setWeight(loss.hessian(instance, previousPredictions[i]));
            xgBoostData.add(xgBoostInstance);
        }
        classifier.buildClassifier(xgBoostData);
        for (int i = 0; i < xgBoostData.numInstances(); i++) {
            previousPredictions[i] += classifier.classifyInstance(xgBoostData.instance(i));
        }
        xgBoostData.delete();
    }

    /** Method that builds the classifier based on the given training data. */
    public void buildClassifier(Instances data) throws Exception {
        super.buildClassifier(data);
        Random random = new Random(getSeed());
        if (data.classAttribute().isNominal()) { loss = new LogLoss(); } else { loss = new SquaredError(); }
        double[] previousPredictions = new double[data.numInstances()];
        xgBoostData = new Instances(data, data.numInstances());
        xgBoostData.replaceAttributeAt(new Attribute("gradient"), data.classIndex());
        initialClassifier = new ZeroR();
        buildBaseClassifier(initialClassifier, data, previousPredictions, xgBoostData);
        for (int j = 0; j < getNumIterations(); j++) {
            ((Randomizable)m_Classifiers[j]).setSeed(random.nextInt());
            buildBaseClassifier(m_Classifiers[j], data, previousPredictions, xgBoostData);
        }
    }

    /** Returns the estimated class probabilities for classification and the numeric prediction for regression. */
    public double[] distributionForInstance(Instance instance) throws Exception {
        Instance xgBoostInstance = (Instance)instance.copy();
        xgBoostInstance.setDataset(xgBoostData);
        double pred = initialClassifier.classifyInstance(xgBoostInstance);
        for (int j = 0; j < getNumIterations(); j++) { pred += m_Classifiers[j].classifyInstance(xgBoostInstance); }
        pred = loss.prediction(pred);
        if (loss instanceof LogLoss) {
            double[] dist = {1.0 - pred, pred}; return dist;
        } else {
            double[] dist = {pred}; return dist;
        }
    }

    /** Returns a description of the ensemble classifier as a string.*/
    public String toString() {
        if (m_Classifiers == null) { return "XGBoost: No model built yet."; }
        StringBuffer text = new StringBuffer();
        text.append("XGBoost with " + getNumIterations() + " iterations and base learner\n\n" + getClassifierSpec());
        text.append("\n\nInitial classifier: \n\n" + initialClassifier.toString());
        text.append("\n\nAll the base classifiers: \n\n");
        for (int i = 0; i < m_Classifiers.length; i++)
            text.append(m_Classifiers[i].toString() + "\n\n");
        return text.toString();
    }
}