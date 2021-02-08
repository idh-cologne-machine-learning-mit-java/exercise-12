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
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration.Builder;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration.ListBuilder;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.Upsampling1D;
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

		// 7 Starting classes
		int internalSize = numInputs*numInputs*outputNum;
		// create the network layout
		ListBuilder nnBuilder = new NeuralNetConfiguration
				.Builder()
				.seed(seed)
//			    .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
				.list();
		
		// Expand two numInputs*numInputs
		nnBuilder.layer(
			new DenseLayer.Builder()
				.nIn(numInputs)
				.nOut(internalSize)
				.build()
		);

		
		nnBuilder.layer(
			new DenseLayer.Builder()
				.nIn(internalSize)
				.nOut(internalSize)
				.activation(Activation.TANH)
				.build()
		);
		
		nnBuilder.layer(
		new DenseLayer.Builder()
				.nIn(internalSize)
				.nOut(internalSize)
				.activation(Activation.ELU)
				.build()
		);

		// outputlayer
		nnBuilder.layer(
			new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
			.activation(Activation.SOFTMAX)
			.nIn(internalSize)
			.nOut(outputNum)
			.build()
		);
		
		MultiLayerConfiguration conf = nnBuilder.build();

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
