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
		String modelWeights = "D:\\OpenCV\\yolov3-tiny.weights";
		String modelConfiguration = "D:\\OpenCV\\yolov3-tiny.cfg";
		
		
		VideoCapture cap = new VideoCapture();
		cap.open(filePath);
		
		Net net = Dnn.readNetFromDarknet(modelConfiguration, modelWeights);
		
		JFrame jframe = new JFrame("Video"); 
		jframe.setContentPane(vidpanel);
		jframe.setSize(1920, 1080);
		jframe.setVisible(true);

		
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
		timer.scheduleAtFixedRate(frameGrabber, 0, 33, TimeUnit.MILLISECONDS); //process each frame at frame rate of vid
	}

	private static List<String> getOutputNames(Net net) { //not mine
		List<String> names = new ArrayList<>();
		List<Integer> outLayers = net.getUnconnectedOutLayers().toList();
		List<String> layersNames = net.getLayerNames();

		outLayers.forEach((item) -> names.add(layersNames.get(item - 1)));
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
		Size sz = new Size(416, 416);
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
		Mat row;
		Mat scores;
		Mat level;
		Core.MinMaxLocResult mm;
		float confidence;
		Point classIdPoint;
		
		Mat blob = Dnn.blobFromImage(frame, 0.00392, sz, new Scalar(0), true);  //edit this maybe, scalar is empty rn, so no mean subtraction i think
		net.setInput(blob);
		net.forward(result, outBlobNames); //result is a 4d tensor, images, height, width, color channels

		float confThreshold = 0.6f; // Insert thresholding beyond which the model will detect objects//

		for (int i = 0; i < result.size(); ++i) {
			// each row is a candidate detection, the 1st 4 numbers are
			// [center_x, center_y, width, height], followed by (N-4) class probabilities
			
			level = result.get(i); //gets i output blob image from network, now a 3d tensor, height, width, color channels
			
			for (int j = 0; j < level.rows(); ++j) {
				row = level.row(j); // gets the data for the image, 
				scores = row.colRange(5, level.cols()); //scores are class probabilities listed after the first 4 values
				mm = Core.minMaxLoc(scores); //finds maximum score in class probabilities
				confidence = (float) mm.maxVal; //confidence = maxVal
				classIdPoint = mm.maxLoc; //Id index is the index of maxVal
				
				if (confidence > confThreshold) {
					centerX = (int) (row.get(0, 0)[0] * frame.cols()); //gets centerX from output blob and scales it for the input image
					centerY = (int) (row.get(0, 1)[0] * frame.rows()); //same but with centerY
					width = (int) (row.get(0, 2)[0] * frame.cols()); //same but with width
					height = (int) (row.get(0, 3)[0] * frame.rows());//same but with height
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
			Dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThresh, indices); //eliminates weaker classifications of same objects
			
			System.out.println(clsIds.get(0));
			int[] ind = indices.toArray();
			int j = 0;
			for (int i = 0; i < ind.length; ++i) {
				int idx = ind[i];
				Rect2d box = boxesArray[idx];
				Imgproc.rectangle(frame, box.tl(), box.br(), new Scalar(255, 255, 0), 2);
			}
			//drawLargestBox(boxesArray, ind, frame); //only biggest object drawn
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


	private static BufferedImage Mat2BufferedImage(Mat original) { //not mine, from openCV tutorial, converts a Mat into a Buff Image
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