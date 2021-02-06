package de.ukoeln.idh.teaching.jml.ex12;

import java.io.File;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.records.reader.impl.transform.TransformProcessRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.transform.TransformProcess;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.transform.transform.categorical.CategoricalToIntegerTransform;
import org.datavec.api.transform.transform.column.RemoveColumnsTransform;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.lossfunctions.LossFunctions;

public class Titanic {

	public static void main(String[] args) throws Exception {
		// the data file
		File datafile = new File("src/main/resources/titanic.csv");

		// a record reader
		RecordReader reader = new CSVRecordReader(1, ',', '"');
		reader.initialize(new FileSplit(datafile));

		// the schema
		Schema dataSchema = new Schema.Builder().addColumnsInteger("PassengerId", "Survived", "Pclass")
				.addColumnsString("Name").addColumnCategorical("Sex", "male", "female", "").addColumnInteger("Age")
				.addColumnsInteger("SipSp", "Parch").addColumnsString("Ticket").addColumnDouble("Fare")
				.addColumnsString("Cabin").addColumnCategorical("Embarked", "C", "S", "Q", "").build();

		// transformation: remove string and unhelpful columns, transform
		// categories into int values
		TransformProcess tp = new TransformProcess.Builder(dataSchema)
				.transform(new RemoveColumnsTransform("PassengerId", "Name", "Ticket", "Cabin"))
				.transform(new CategoricalToIntegerTransform("Sex"))
				.transform(new CategoricalToIntegerTransform("Embarked")).build();

		// Create a reader to read from the transform process
		RecordReader procReader = new TransformProcessRecordReader(reader, tp);

		// we are only interested in the first data set
		DataSetIterator trainDataSetIterator = new RecordReaderDataSetIterator(procReader, 892, 0, 2);
		DataSet allData = trainDataSetIterator.next();
		allData.shuffle();

		// split into training and test
		SplitTestAndTrain testAndTrain = allData.splitTestAndTrain(0.65);
		DataSet trainingData = testAndTrain.getTrain();
		DataSet testData = testAndTrain.getTest();

		// number of input features
		final int numInputs = tp.getFinalSchema().numColumns() - 1;

		// number of output categories
		int outputNum = allData.numOutcomes();

		// seed value for the RNG, fixed for reproducibility
		long seed = 6;

		// create the network layout
		/*
		 * MultiLayerConfiguration conf = new
		 * NeuralNetConfiguration.Builder().seed(seed).list() .layer(new
		 * DenseLayer.Builder().nIn(numInputs).nOut(100).build()) .layer(new
		 * DenseLayer.Builder().nIn(100).nOut(100).build()) .layer(new
		 * DenseLayer.Builder().nIn(100).nOut(100).build()) .layer(new
		 * OutputLayer.Builder(LossFunctions.LossFunction.MCXENT).activation(Activation.
		 * SOFTMAX) .nIn(100).nOut(outputNum).build()) .build();
		 */
		
		 // create the network layout (toggle block comment and uncomment ll. 74-81 in order to inactivate this)
	    MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().seed(seed)
	        .activation(Activation.TANH)
	        .list()
	        .layer(new DenseLayer.Builder().nIn(numInputs).nOut(10).build())
	        .layer(new DenseLayer.Builder().nIn(10).nOut(400).build())
	        .layer(new DenseLayer.Builder().nIn(400).nOut(100).build())
	        .layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD).activation(Activation.SOFTMAX)
	            .nIn(100).nOut(outputNum).build())
	        .build();

		// Create model
		MultiLayerNetwork model = new MultiLayerNetwork(conf);
		model.init();
		model.setListeners(new ScoreIterationListener(10));

		// start training of 1000 epochs
		for (int i = 0; i < 1000; i++) {
			model.fit(trainingData);
		}

		// evaluate the model on the test set
		Evaluation eval = new Evaluation(2);
		INDArray output = model.output(testData.getFeatures());
		eval.eval(testData.getLabels(), output);
		System.out.println(eval.stats());
	}

}
