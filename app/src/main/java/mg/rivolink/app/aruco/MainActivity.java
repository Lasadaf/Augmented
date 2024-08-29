package mg.rivolink.app.aruco;

import android.app.Activity;
import android.app.AlertDialog;
import android.content.Intent;
import android.content.DialogInterface;

import android.graphics.PixelFormat;
import android.opengl.GLSurfaceView;
import android.os.Bundle;
import android.support.v7.app.AppCompatActivity;

import android.view.View;
import android.view.WindowManager;
import android.widget.TextView;
import android.widget.Toast;

import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;

import mg.rivolink.app.aruco.utils.CameraParameters;
import mg.rivolink.app.aruco.view.LandscapeCameraLayout;
import mg.rivolink.app.aruco.view.PortraitCameraLayout;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener2;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.aruco.Aruco;
import org.opencv.aruco.DetectorParameters;
import org.opencv.aruco.Dictionary;
import org.opencv.calib3d.Calib3d;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDouble;
import org.opencv.core.MatOfInt;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.MatOfPoint3f;
import org.opencv.core.Point;
import org.opencv.core.Point3;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;

public class MainActivity extends AppCompatActivity implements CvCameraViewListener2 {

	public static final float SIZE = 0.04f;

	private Mat cameraMatrix;
	private MatOfDouble distCoeffs;

	private Mat rgb;
	private Mat gray;

	private Mat rvecs;
	private Mat tvecs;

	private MatOfInt ids;
	private List<Mat> corners;
	private Dictionary dictionary;
	private DetectorParameters parameters;

	private CameraBridgeViewBase camera;

	private TextView text;
	private int frame_i = 0;
	private double min_x= -1, min_y = -1, max_x = -1, max_y = -1;

	private GLSurfaceView glSurfaceView;
	private OpenGLRenderer glRenderer;

	private final BaseLoaderCallback loaderCallback = new BaseLoaderCallback(this){
        @Override
        public void onManagerConnected(int status){
			if(status == LoaderCallbackInterface.SUCCESS){
				Activity activity = MainActivity.this;

				cameraMatrix = Mat.eye(3, 3, CvType.CV_64FC1);
				distCoeffs = new MatOfDouble(Mat.zeros(5, 1, CvType.CV_64FC1));

				if(CameraParameters.fileExists(activity)){
					CameraParameters.tryLoad(activity, cameraMatrix, distCoeffs);
				}
				else {
					CameraParameters.selectFile(activity);
				}

				camera.enableView();
			}
			else {
				super.onManagerConnected(status);
			}
        }
    };

    @Override
    protected void onCreate(Bundle savedInstanceState){
        super.onCreate(savedInstanceState);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        setContentView(R.layout.main_layout);

        camera = ((LandscapeCameraLayout) findViewById(R.id.camera)).getCamera();
        camera.setCvCameraViewListener(this);

		glSurfaceView = (GLSurfaceView) findViewById(R.id.text2);
		glSurfaceView.setZOrderOnTop(true);
		glSurfaceView.setEGLConfigChooser(8, 8, 8, 8, 16, 0);
		glSurfaceView.getHolder().setFormat(PixelFormat.TRANSLUCENT);
		glSurfaceView.setEGLContextClientVersion(2);
		glRenderer = new OpenGLRenderer(this);
		glSurfaceView.setRenderer(glRenderer);

		//text = (TextView) findViewById(R.id.text);
	}

	@Override
	protected void onActivityResult(int requestCode, int resultCode, Intent data){
		CameraParameters.onActivityResult(this, requestCode, resultCode, data, cameraMatrix, distCoeffs);
	}

	@Override
    public void onResume(){
        super.onResume();

		if(OpenCVLoader.initDebug())
			loaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
		else
			Toast.makeText(this, getString(R.string.error_native_lib), Toast.LENGTH_LONG).show();
		glSurfaceView.onResume();
    }

	@Override
    public void onPause(){
        super.onPause();

        if(camera != null)
            camera.disableView();
		glSurfaceView.onPause();
    }

	@Override
    public void onDestroy(){
        super.onDestroy();

        if (camera != null)
            camera.disableView();
    }

	@Override
	public void onCameraViewStarted(int width, int height){
		rgb = new Mat();
		corners = new LinkedList<>();
		parameters = DetectorParameters.create();
		dictionary = Aruco.getPredefinedDictionary(Aruco.DICT_6X6_50);
	}

	@Override
	public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame){
		if(!CameraParameters.isLoaded()){
			return inputFrame.rgba();
		}

		Imgproc.cvtColor(inputFrame.rgba(), rgb, Imgproc.COLOR_RGBA2RGB);
		gray = inputFrame.gray();

		ids = new MatOfInt();
		corners.clear();

		Aruco.detectMarkers(gray, dictionary, corners, ids, parameters);

		if(corners.size()>0){
			Aruco.drawDetectedMarkers(rgb, corners, ids);

			rvecs = new Mat();
			tvecs = new Mat();

			Aruco.estimatePoseSingleMarkers(corners, SIZE, cameraMatrix, distCoeffs, rvecs, tvecs);

			List<Point3> points = new ArrayList<Point3>();
			points.add(new Point3(-0.02f, -0.02f, 0));
			points.add(new Point3(-0.02f,  0.02f, 0));
			points.add(new Point3( 0.02f,  0.02f, 0));
			points.add(new Point3( 0.02f, -0.02f, 0));
			points.add(new Point3(-0.02f, -0.02f, SIZE));
			points.add(new Point3(-0.02f,  0.02f, SIZE));
			points.add(new Point3( 0.02f,  0.02f, SIZE));
			points.add(new Point3( 0.02f, -0.02f, SIZE));

			MatOfPoint3f cubePoints = new MatOfPoint3f();
			cubePoints.fromList(points);

			MatOfPoint2f projectedPoints = new MatOfPoint2f();

			Calib3d.projectPoints(cubePoints, rvecs.row(0), tvecs.row(0), cameraMatrix, distCoeffs, projectedPoints);

			List<Point> pts = projectedPoints.toList();

			//for(int i=0; i<4; i++){
			//	Imgproc.line(rgb, pts.get(i), pts.get((i+1)%4), new Scalar(255, 0, 0), 2);
			//	Imgproc.line(rgb, pts.get(i+4), pts.get(4+(i+1)%4), new Scalar(255, 0, 0), 2);
			//	Imgproc.line(rgb, pts.get(i), pts.get(i+4), new Scalar(255, 0, 0), 2);
			//}

			runOnUiThread(new Runnable() {
				@Override
				public void run() {
					glSurfaceView.setVisibility(View.VISIBLE);
				}
			});

			float[] v = new float[16];
			for (int i = 0 ; i < 8 ; i++) {
				v[2 * i] = (float) ((pts.get(i).x - 960.0) / 960.0);
				v[2 * i + 1] = (float) ((pts.get(i).y - 540.0) / -540.0);
			}

			float front_z, back_z, left_z, right_z;

			if (v[1] + v[7] < v[3] + v[5]) {
				front_z = 0.25f;
				back_z = 0.1f;
			} else {
				back_z = 0.25f;
				front_z = 0.1f;
			}

			if (v[1] + v[3] < v[5] + v[7]) {
				left_z = 0.25f;
				right_z = 0.1f;
			} else {
				right_z = 0.25f;
				left_z = 0.1f;
			}

			//String mes = "front: ";
			//mes += (v[1] + v[7]) + ", front_z: " + front_z + "; left:" + (v[1] + v[3]) + ", left_z: " + left_z + "; back:" + (v[3] + v[5]) + ", back_z: " + back_z + "; right:" + (v[5] + v[7]) + ", right_z: " + right_z;
			//text.setText(mes);

			float[] lines = {
					v[0], v[1],
					v[2], v[3],
					v[2], v[3],
					v[4], v[5],
					v[4], v[5],
					v[6], v[7],
					v[6], v[7],
					v[0], v[1],
					v[8], v[9],
					v[10], v[11],
					v[10], v[11],
					v[12], v[13],
					v[12], v[13],
					v[14], v[15],
					v[14], v[15],
					v[8], v[9],
					v[0], v[1],
					v[8], v[9],
					v[2], v[3],
					v[10], v[11],
					v[4], v[5],
					v[12], v[13],
					v[6], v[7],
					v[14], v[15]
			};

			glRenderer.vertices = lines;
		} else {
			runOnUiThread(new Runnable() {
				@Override
				public void run() {
				glSurfaceView.setVisibility(View.INVISIBLE);
				}
			});
		}

		return rgb;
	}

	@Override
	public void onCameraViewStopped(){
		rgb.release();
	}

}


