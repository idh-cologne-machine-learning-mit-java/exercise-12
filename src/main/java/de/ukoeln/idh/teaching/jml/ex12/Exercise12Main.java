package de.ukoeln.idh.teaching.jml.ex12;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.StreamTokenizer;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;

import org.apache.commons.compress.compressors.bzip2.BZip2CompressorInputStream;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.transform.TransformProcessRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.transform.TransformProcess;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.transform.transform.categorical.CategoricalToIntegerTransform;
import org.datavec.api.transform.transform.categorical.StringToCategoricalTransform;
import org.datavec.api.transform.transform.string.ChangeCaseStringTransform;
import org.datavec.api.transform.transform.string.ReplaceStringTransform;
import org.datavec.api.transform.transform.string.StringListToCountsNDArrayTransform;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.lossfunctions.LossFunctions;

public class Exercise12Main {

	public static void main(String[] args) throws Exception {
		File trainFile = new File("src/main/resources/amazon/train_small.ft.txt.bz2");
		File testFile = new File("src/main/resources/amazon/test_small.ft.txt.bz2");

		MyRecordReader trainReader = new MyRecordReader();
		trainReader.initialize(new FileSplit(trainFile));
		MyRecordReader testReader = new MyRecordReader();
		testReader.initialize(new FileSplit(testFile));

		Schema dataSchema = new Schema.Builder().addColumnString("label").addColumnString("text").build();

		List<String> vocabulary = readVocabulary(testFile, 15);
		Map<String, String> map = new HashMap<String, String>();
		map.put("\\d", " ");
		map.put("\\p{Punct}", " ");

		TransformProcess tp = new TransformProcess.Builder(dataSchema)
				.transform(new ChangeCaseStringTransform("text", ChangeCaseStringTransform.CaseType.LOWER))
				.transform(new ReplaceStringTransform("text", map))
				.transform(new StringListToCountsNDArrayTransform("text", vocabulary, " ", false, true))
				.transform(new StringToCategoricalTransform("label", "__label__2", "__label__1"))
				.transform(new CategoricalToIntegerTransform("label")).build();

		RecordReader procTrainReader = new TransformProcessRecordReader(trainReader, tp);
		RecordReader procTestReader = new TransformProcessRecordReader(testReader, tp);
		DataSetIterator trainDataSetIterator = new RecordReaderDataSetIterator(procTrainReader, 5, 0, 2);
		DataSetIterator testDataSetIterator = new RecordReaderDataSetIterator(procTestReader, 5, 0, 2);

		int rngSeed = 123;
		int numColumns = vocabulary.size();

		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
				.seed(rngSeed).list()
				.layer(new DenseLayer.Builder()
						.nIn(numColumns).nOut(200).activation(Activation.SIGMOID)
						.build())
				.layer(new DenseLayer.Builder()
						.nIn(200).nOut(100).activation(Activation.TANH)
						.build())
				.layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
						.nIn(100).nOut(2).activation(Activation.SOFTMAX).build())
				.validateOutputLayerConfig(true).build();
		MultiLayerNetwork network = new MultiLayerNetwork(conf);
		network.init();

		int eachIterations = 100;
		network.addListeners(new ScoreIterationListener(eachIterations));

		network.fit(trainDataSetIterator);

		Evaluation eval = network.evaluate(testDataSetIterator);

		System.out.println("Loss " + network.score());
		System.out.println("Accuracy: " + eval.accuracy());
		System.out.println("Precision: " + eval.precision());
		System.out.println("Recall: " + eval.recall());

	}

	public static List<String> readVocabulary(File file, int limit) throws FileNotFoundException, IOException {
		Map<String, Integer> vocab = new HashMap<String, Integer>();
		System.err.println("Reading vocabulary ... ");
		try (InputStreamReader reader = new InputStreamReader(
				new BZip2CompressorInputStream(new FileInputStream(file)))) {

			StreamTokenizer tokenizer = new StreamTokenizer(reader);
			tokenizer.slashSlashComments(false);
			tokenizer.slashStarComments(false);
			tokenizer.lowerCaseMode(false);
			int currentToken = tokenizer.nextToken();
			while (currentToken != StreamTokenizer.TT_EOF) {
				if (tokenizer.ttype == StreamTokenizer.TT_WORD) {
					String w = tokenizer.sval;
					if (vocab.containsKey(w))
						vocab.put(w, vocab.get(w) + 1);
					else
						vocab.put(w, 1);
				}
				currentToken = tokenizer.nextToken();
			}
		}
		LinkedList<String> finalVocab = new LinkedList<String>();
		for (String w : vocab.keySet()) {
			if (vocab.get(w) >= limit)
				finalVocab.add(w);

		}

		System.err.println("Done. Vocabulary contains " + finalVocab.size() + " words.");
		return finalVocab;
	}
}
