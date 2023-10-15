package weka.classifiers.trees;

import weka.classifiers.Classifier;
import weka.classifiers.RandomizableClassifier;
import weka.core.*;

import java.io.Serializable;
import java.util.*;
import java.util.stream.IntStream;
import java.util.stream.Collectors;

/**
 * Basic WEKA classifiers (and this includes learning algorithms that build regression models!)
 * should simply extend AbstractClassifier but this classifier is also randomizable.
 */
public class XGBoostTree extends RandomizableClassifier implements WeightedInstancesHandler, AdditionalMeasureProducer {

    /**
     * Provides an enumeration consisting of a single element: "measureNumRules".
     */
    public Enumeration<String> enumerateMeasures() {
        String[] measures = {"measureNumRules"};
        return Collections.enumeration(Arrays.asList(measures));
    }

    /**
     * Provides the number of leaves for "measureNumRules" and throws an exception for other arguments.
     */
    public double getMeasure(String measureName) throws IllegalArgumentException {
        if (measureName.equals("measureNumRules")) {
            return getNumLeaves(rootNode);
        } else {
            throw new IllegalArgumentException("Measure " + measureName + " not supported.");
        }
    }

    /**
     * The hyperparameters for an XGBoost tree.
     */
    private double eta = 0.3;

    @OptionMetadata(displayName = "eta", description = "eta",
            commandLineParamName = "eta", commandLineParamSynopsis = "-eta <double>", displayOrder = 1)
    public void setEta(double e) { eta = e; }

    public double getEta() { return eta; }

    private double lambda = 1.0;

    @OptionMetadata(displayName = "lambda", description = "lambda",
            commandLineParamName = "lambda", commandLineParamSynopsis = "-lambda <double>", displayOrder = 2)
    public void setLambda(double l) { lambda = l; }

    public double getLambda() { return lambda; }

    private double gamma = 1.0;

    @OptionMetadata(displayName = "gamma", description = "gamma",
            commandLineParamName = "gamma", commandLineParamSynopsis = "-gamma <double>", displayOrder = 3)
    public void setGamma(double l) { gamma = l; }

    public double getGamma() { return gamma; }

    private double subsample = 0.5;

    @OptionMetadata(displayName = "subsample", description = "subsample",
            commandLineParamName = "subsample", commandLineParamSynopsis = "-subsample <double>", displayOrder = 4)
    public void setSubsample(double s) { subsample = s; }

    public double getSubsample() { return subsample; }

    private double colsample_bynode = 0.5;

    @OptionMetadata(displayName = "colsample_bynode", description = "colsample_bynode",
            commandLineParamName = "colsample_bynode", commandLineParamSynopsis = "-colsample_bynode <double>", displayOrder = 5)
    public void setColSampleByNode(double c) { colsample_bynode = c; }

    public double getColSampleByNode() { return colsample_bynode; }

    private int max_depth = 6;

    @OptionMetadata(displayName = "max_depth", description = "max_depth",
            commandLineParamName = "max_depth", commandLineParamSynopsis = "-max_depth <int>", displayOrder = 6)
    public void setMaxDepth(int m) { max_depth = m; }

    public int getMaxDepth() { return max_depth; }

    private double min_child_weight = 1.0;

    @OptionMetadata(displayName = "min_child_weight", description = "min_child_weight",
            commandLineParamName = "min_child_weight", commandLineParamSynopsis = "-min_child_weight <double>", displayOrder = 7)
    public void setMinChildWeight(double w) { min_child_weight = w; }

    public double getMinChildWeight() { return min_child_weight; }

    /**
     * A possible way to represent the tree structure using Java records.
     */
    private interface Node { }

    private record InternalNode(Attribute attribute, double splitPoint, Node leftSuccessor, Node rightSuccessor)
            implements Node, Serializable { }

    private record LeafNode(double prediction) implements Node, Serializable { }

    /**
     * The root node of the decision tree.
     */
    private Node rootNode = null;

    /**
     * The training instances.
     */
    private Instances data;

    /**
     * Random number generator to be used for subsampling rows and columns.
     */
    Random random;

    /**
     * A class for objects that hold a split specification, including the quality of the split.
     */
    private class SplitSpecification {
        private final Attribute attribute;
        private double splitPoint;
        private double splitQuality;

        private SplitSpecification(Attribute attribute, double splitQuality, double splitPoint) {
            this.attribute = attribute;
            this.splitQuality = splitQuality;
            this.splitPoint = splitPoint;
        }
    }

    /**
     * A class for objects that contain the sufficient statistics required to measure split quality.
     */
    private class SufficientStatistics {
        private double sumOfNegativeGradients = 0.0;
        private double sumOfHessians = 0.0;

        private SufficientStatistics(double sumOfNegativeGradients, double sumOfHessians) {
            this.sumOfNegativeGradients = sumOfNegativeGradients;
            this.sumOfHessians = sumOfHessians;
        }

        private void updateStats(double negativeGradient, double hessian, boolean add) {
            sumOfNegativeGradients = (add) ? sumOfNegativeGradients +
                    negativeGradient : sumOfNegativeGradients - negativeGradient;
            sumOfHessians = (add) ? sumOfHessians + hessian : sumOfHessians - hessian;
        }
    }

    /**
     * Computes the "impurity" for a subset of data.
     */
    private double impurity(SufficientStatistics ss) {
        return (ss.sumOfHessians <= 0.0) ? 0.0 :
                ss.sumOfNegativeGradients * ss.sumOfNegativeGradients / (ss.sumOfHessians + lambda);
    }

    /**
     * Computes the reduction in the sum of squared errors based on the sufficient statistics provided. The
     * variable i holds the sufficient statistics based on the data before it is split,
     * the variable l holds the sufficient statistics for the left branch, and the variable r hold the sufficient
     * statistics for the right branch.
     */
    private double splitQuality(SufficientStatistics i, SufficientStatistics l, SufficientStatistics r) {
        return 0.5 * (impurity(l) + impurity(r) - impurity(i)) - gamma;
    }

    /**
     * Finds the best split point and returns the corresponding split specification object. The given indices
     * define the subset of the training set for which the split is to be found. The initialStats are the sufficient
     * statistics before the data is split.
     */
    private SplitSpecification findBestSplitPoint(int[] indices, Attribute attribute, SufficientStatistics initialStats) {
        var statsLeft = new SufficientStatistics(0.0, 0.0);
        var statsRight = new SufficientStatistics(initialStats.sumOfNegativeGradients, initialStats.sumOfHessians);
        var splitSpecification = new SplitSpecification(attribute, 1e-6, Double.NEGATIVE_INFINITY);
        var previousValue = Double.NEGATIVE_INFINITY;
        for (int i : Arrays.stream(Utils.sortWithNoMissingValues(Arrays.stream(indices).mapToDouble(x ->
                data.instance(x).value(attribute)).toArray())).map(x -> indices[x]).toArray()) {
            Instance instance = data.instance(i);
            if (instance.value(attribute) > previousValue) {
                if (statsLeft.sumOfHessians != 0 && statsRight.sumOfHessians != 0 &&
                        statsLeft.sumOfHessians >= min_child_weight && statsRight.sumOfHessians >= min_child_weight) {
                    var splitQuality = splitQuality(initialStats, statsLeft, statsRight);
                    if (splitQuality > splitSpecification.splitQuality) {
                        splitSpecification.splitQuality = splitQuality;
                        splitSpecification.splitPoint = (instance.value(attribute) + previousValue) / 2.0;
                    }
                }
                previousValue = instance.value(attribute);
            }
            statsLeft.updateStats(instance.classValue(), instance.weight(), true);
            statsRight.updateStats(instance.classValue(), instance.weight(), false);
        }
        return splitSpecification;
    }

    /**
     * Recursively grows a tree for a subset of data specified by the given indices.
     */
    private Node makeTree(int[] indices, int depth) {
        var stats = new SufficientStatistics(0.0, 0.0);
        for (int i : indices) {
            stats.updateStats(data.instance(i).classValue(), data.instance(i).weight(), true);
        }
        if (stats.sumOfHessians <= 0.0 || stats.sumOfHessians < min_child_weight || depth >= max_depth) {
            return new LeafNode(eta * stats.sumOfNegativeGradients / (stats.sumOfHessians + lambda));
        }
        var bestSplitSpecification = new SplitSpecification(null, Double.NEGATIVE_INFINITY, Double.NEGATIVE_INFINITY);
        List<Integer> attributes = new ArrayList<>(this.data.numAttributes() - 1);
        for (int i = 0; i < data.numAttributes(); i++) {
            if (i != this.data.classIndex()) {
                attributes.add(i);
            }
        }
        if (colsample_bynode < 1.0) {
            Collections.shuffle(attributes, random);
        }
        for (Integer index : attributes.subList(0, (int) (colsample_bynode * attributes.size()))) {
            var splitSpecification = findBestSplitPoint(indices, data.attribute(index), stats);
            if (splitSpecification.splitQuality > bestSplitSpecification.splitQuality) {
                bestSplitSpecification = splitSpecification;
            }
        }
        if (bestSplitSpecification.splitQuality <= 1e-6) {
            return new LeafNode(eta * stats.sumOfNegativeGradients / (stats.sumOfHessians + lambda));
        } else {
            var leftSubset = new ArrayList<Integer>(indices.length);
            var rightSubset = new ArrayList<Integer>(indices.length);
            for (int i : indices) {
                if (data.instance(i).value(bestSplitSpecification.attribute) < bestSplitSpecification.splitPoint) {
                    leftSubset.add(i);
                } else {
                    rightSubset.add(i);
                }
            }
            return new InternalNode(bestSplitSpecification.attribute, bestSplitSpecification.splitPoint,
                    makeTree(leftSubset.stream().mapToInt(Integer::intValue).toArray(), depth + 1),
                    makeTree(rightSubset.stream().mapToInt(Integer::intValue).toArray(), depth + 1));
        }
    }

    /**
     * Returns the capabilities of the classifier: numeric predictors and numeric target.
     */
    public Capabilities getCapabilities() {
        Capabilities result = super.getCapabilities();
        result.disableAll();
        result.enable(Capabilities.Capability.NUMERIC_ATTRIBUTES);
        result.enable(Capabilities.Capability.NUMERIC_CLASS);
        return result;
    }

    /**
     * Builds the tree model by calling the recursive makeTree(Instances) method.
     */
    public void buildClassifier(Instances trainingData) throws Exception {
        getCapabilities().testWithFail(trainingData);
        random = new Random(getSeed());
        this.data = new Instances(trainingData);
        if (subsample < 1.0) {
            this.data.randomize(random);
        }
        this.data = new Instances(this.data, 0, (int) (subsample * this.data.numInstances()));
        rootNode = makeTree(IntStream.range(0, this.data.numInstances()).toArray(), 0);
        data = null;
        random = null;
    }

    /**
     * Recursive method for obtaining a prediction from the tree attached to the node provided.
     */
    private double makePrediction(Node node, Instance instance) {
        if (node instanceof LeafNode) {
            return ((LeafNode) node).prediction;
        } else if (node instanceof InternalNode) {
            if (instance.value(((InternalNode) node).attribute) < ((InternalNode) node).splitPoint) {
                return makePrediction(((InternalNode) node).leftSuccessor, instance);
            } else {
                return makePrediction(((InternalNode) node).rightSuccessor, instance);
            }
        }
        return Utils.missingValue(); // This should never happen
    }

    /**
     * Provides a prediction for the current instance by calling the recursive makePrediction(Node, Instance) method.
     */
    public double classifyInstance(Instance instance) {
        return makePrediction(rootNode, instance);
    }

    /**
     * Returns the number of leaves in the tree.
     */
    public int getNumLeaves(Node node) {
        if (node instanceof LeafNode) {
            return 1;
        } else {
            return getNumLeaves(((InternalNode)node).leftSuccessor) + getNumLeaves(((InternalNode)node).rightSuccessor);
        }
    }

    /**
     * Recursively produces the string representation of a branch in the tree.
     */
    private void branchToString(StringBuffer sb, boolean left, int level, InternalNode node) {
        sb.append("\n");
        for (int j = 0; j < level; j++) {
            sb.append("|   ");
        }
        sb.append(node.attribute.name() + (left ? " < " : " >= ") + Utils.doubleToString(node.splitPoint, getNumDecimalPlaces()));
        toString(sb, level + 1, left ? node.leftSuccessor : node.rightSuccessor);
    }

    /**
     * Recursively produces a string representation of a subtree by calling the branchToString(StringBuffer, int,
     * Node) method for both branches, unless we are at a leaf.
     */
    private void toString(StringBuffer sb, int level, Node node) {
        if (node instanceof LeafNode) {
            sb.append(": " + Utils.doubleToString(((LeafNode) node).prediction, getNumDecimalPlaces()));
        } else {
            branchToString(sb, true, level, (InternalNode) node);
            branchToString(sb, false, level, (InternalNode) node);
        }
    }

    /**
     * Returns a string representation of the tree by calling the recursive toString(StringBuffer, int, Node) method.
     */
    public String toString() {
        StringBuffer sb = new StringBuffer();
        toString(sb, 0, rootNode);
        return sb.toString();
    }

    /**
     * The main method for running this classifier from a command-line interface.
     */
    public static void main(String[] options) {
        runClassifier(new XGBoostTree(), options);
    }
}