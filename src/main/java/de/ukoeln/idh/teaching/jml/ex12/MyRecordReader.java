package de.ukoeln.idh.teaching.jml.ex12;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.net.URI;
import java.util.ArrayList;
import java.util.List;

import org.apache.commons.compress.compressors.bzip2.BZip2CompressorInputStream;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.writable.Text;
import org.datavec.api.writable.Writable;
import org.nd4j.common.function.Function;

public class MyRecordReader extends CSVRecordReader {

	private static final long serialVersionUID = 1L;

	public MyRecordReader() {
		streamCreatorFn = new Function<URI, InputStream>() {

			@Override
			public InputStream apply(URI t) {
				String location = t.toString();
				FileInputStream ret;
				try {
					ret = location.startsWith("file:") ? new FileInputStream(new File(URI.create(location)))
							: new FileInputStream(new File(location));
					return new BZip2CompressorInputStream(ret);
				} catch (IOException e) {
					e.printStackTrace();
				}
				return null;
			}
		};
	}

	@Override
	protected List<Writable> parseLine(String line) {
		List<Writable> ret = new ArrayList<>();
		String label = line.substring(0, 10);
		String text = line.substring(11);
		ret.add(new Text(label));
		ret.add(new Text(text));
		return ret;
	}
}
