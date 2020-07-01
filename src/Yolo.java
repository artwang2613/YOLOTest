
import org.opencv.core.*;
import org.opencv.dnn.*;
import org.opencv.imgproc.Imgproc;
import org.opencv.utils.*;

import org.opencv.videoio.VideoCapture;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;

import javax.swing.ImageIcon;
import javax.swing.JFrame;
import javax.swing.JLabel;

public class Yolo {

	private static String filePath = "D:\\DeepLearningDatasets\\People\\DadSonWalking.mp4";
	private static JLabel vidpanel = new JLabel();
	private static ScheduledExecutorService timer;

	public static void main(String[] args) throws InterruptedException {
		System.load("D:\\OpenCV\\opencv\\build\\java\\x64\\opencv_java430.dll");
		VideoCapture cap = new VideoCapture(); // Load video using the videocapture OpenCV class//
		cap.open(filePath);
		String modelWeights = "D:\\OpenCV\\yolov3-tiny.weights";
		String modelConfiguration = "D:\\OpenCV\\yolov3-tiny.cfg";
		Net net = Dnn.readNetFromDarknet(modelConfiguration, modelWeights);
		JFrame jframe = new JFrame("Video"); 

		jframe.setContentPane(vidpanel);
		jframe.setSize(1920, 1080);
		jframe.setVisible(true);// we instantiate the frame here//

		Runnable frameGrabber = new Runnable() {

			@Override
			public void run() {
				Mat frame = grabFrame(cap, net);
				ImageIcon imageToShow = new ImageIcon(Mat2BufferedImage(frame));
				vidpanel.setIcon(imageToShow);
				vidpanel.repaint();
			}
		};
		timer = Executors.newSingleThreadScheduledExecutor();
		timer.scheduleAtFixedRate(frameGrabber, 0, 33, TimeUnit.MILLISECONDS);
	}

	private static List<String> getOutputNames(Net net) { //not mine
		List<String> names = new ArrayList<>();

		List<Integer> outLayers = net.getUnconnectedOutLayers().toList();
		List<String> layersNames = net.getLayerNames();

		outLayers.forEach((item) -> names.add(layersNames.get(item - 1)));// unfold and create R-CNN layers from the
																			// loaded YOLO model//
		return names;
	}

	private static Mat grabFrame(VideoCapture cap, Net net) { //not completely mine, partially from openCV demo
		Mat frame = new Mat();

		if (cap.isOpened()) {
			try {
				cap.read(frame);
				if (!frame.empty()) {
					analyzeFrame(cap, frame, net);
				}
			} catch (Exception e) {
				System.err.println("Exception during the image elaboration: " + e);
			}
		}
		return frame;
	}

	private static void analyzeFrame(VideoCapture cap, Mat frame, Net net) {
		Size sz = new Size(256, 256);

		List<Mat> result = new ArrayList<>();
		List<Integer> clsIds = new ArrayList<>();
		List<Float> confs = new ArrayList<>();
		List<Rect2d> rects = new ArrayList<>();
		List<String> outBlobNames = getOutputNames(net);

		int centerX;
		int centerY;
		int width;
		int height;
		int left;
		int top;
		
		/*clsIds.clear();
		confs.clear();
		rects.clear();*/

		Mat blob = Dnn.blobFromImage(frame, 0.00392, sz, new Scalar(0), true); 
		net.setInput(blob);
		net.forward(result, outBlobNames);

		//outBlobNames.forEach(System.out::println);
		// result.forEach(System.out::println);

		float confThreshold = 0.6f; // Insert thresholding beyond which the model will detect objects//

		for (int i = 0; i < result.size(); ++i) {
			// each row is a candidate detection, the 1st 4 numbers are
			// [center_x, center_y, width, height], followed by (N-4) class probabilities
			Mat level = result.get(i);
			Mat row;// = level.row(j);
			Mat scores;// = row.colRange(5, level.cols());
			Core.MinMaxLocResult mm;// = Core.minMaxLoc(scores);
			float confidence;// = (float) mm.maxVal;
			Point classIdPoint;// = mm.maxLoc;
			for (int j = 0; j < level.rows(); ++j) {
				row = level.row(j);
				scores = row.colRange(5, level.cols());
				mm = Core.minMaxLoc(scores);
				confidence = (float) mm.maxVal;
				classIdPoint = mm.maxLoc;
				if (confidence > confThreshold) {
					centerX = (int) (row.get(0, 0)[0] * frame.cols()); // scaling for drawing the bounding
																		// boxes//
					centerY = (int) (row.get(0, 1)[0] * frame.rows());
					width = (int) (row.get(0, 2)[0] * frame.cols());
					height = (int) (row.get(0, 3)[0] * frame.rows());
					left = centerX - width / 2;
					top = centerY - height / 2;

					clsIds.add((int) classIdPoint.x);
					confs.add((float) confidence);
					rects.add(new Rect2d(left, top, width, height));
				}
			}
		}

		float nmsThresh = 0.5f;
		MatOfFloat confidences;
		if (!confs.isEmpty()) {
			confidences = new MatOfFloat(Converters.vector_float_to_Mat(confs));

			Rect2d[] boxesArray = rects.toArray(new Rect2d[0]);
			MatOfRect2d boxes = new MatOfRect2d();
			boxes.fromArray(boxesArray);
			MatOfInt indices = new MatOfInt();
			Dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThresh, indices); // We draw the bounding
																					// boxes
																					// for
																					// objects here//
			System.out.println(clsIds.get(0));
			int[] ind = indices.toArray();
			int j = 0;
			for (int i = 0; i < ind.length; ++i) {
				int idx = ind[i];
				Rect2d box = boxesArray[idx];
				Imgproc.rectangle(frame, box.tl(), box.br(), new Scalar(255, 255, 0), 2);
			}
			//drawLargestBox(boxesArray, ind, frame);
		}
	}



	private static void drawLargestBox(Rect2d[] boxes, int[] indices, Mat frame) {
		double largestArea = 0;
		Rect2d largestBox = null;
		Rect2d curBox = null;
		for (int i = 0; i < indices.length; ++i) {
			curBox = boxes[indices[i]];
			if(curBox.x >= 400 && curBox.x <= 1520) {
				if (curBox.area() >= largestArea) {
					largestBox = curBox;
					largestArea = curBox.area();
				}
			}
		}
		Imgproc.rectangle(frame, largestBox.tl(), largestBox.br(), new Scalar(255, 200, 0), 2);
	}

//		}
	private static BufferedImage Mat2BufferedImage(Mat original) { // The class described here takes in matrix and
																	// renders
																	// the video to the frame //
		BufferedImage image = null;
		int width = original.width(), height = original.height(), channels = original.channels();
		byte[] sourcePixels = new byte[width * height * channels];
		original.get(0, 0, sourcePixels);

		if (original.channels() > 1) {
			image = new BufferedImage(width, height, BufferedImage.TYPE_3BYTE_BGR);
		} else {
			image = new BufferedImage(width, height, BufferedImage.TYPE_BYTE_GRAY);
		}
		final byte[] targetPixels = ((DataBufferByte) image.getRaster().getDataBuffer()).getData();
		System.arraycopy(sourcePixels, 0, targetPixels, 0, sourcePixels.length);

		return image;
	}
}
