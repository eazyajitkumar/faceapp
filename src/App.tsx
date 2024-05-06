import * as faceapi from "face-api.js";
import React, { useCallback, useEffect, useState } from "react";
// import Webcam from "react-webcam";
import "./App.css";

type ModalType = "TRAINING_IMAGE" | "QUERY_IMAGE";

function App() {
  const [result, setResult] = useState<number | undefined>(undefined);
  const [modalType, setModalType] = useState<ModalType | null>(null);
  const [trainingImages, setTrainingImages] = useState<string[]>([]);
  const [queryImage, setQueryImage] = useState<string>("");
  const [isMatching, setIsMatching] = useState<boolean>(false);
  const [modelsLoaded, setModelsLoaded] = useState<boolean>(false);

  const dialogRef = React.useRef<HTMLDialogElement>();
  const loaderDialogRef = React.useRef<HTMLDialogElement>();
  const webcamRef = React.useRef();

  useEffect(() => {
    sendMessage("App started");
    const messageListener = document.addEventListener(
      "message",
      async (event) => {
        console.log("messageListener: ", event?.data);
        let recievedData = JSON.parse(event?.data);
        let type = recievedData?.type;
        if (type === "MATCH_FACE") {
          let training = recievedData?.trainingImages;
          let query = recievedData?.queryImage;
          try {
            const result = await matchFaces(query, training);
            let output = JSON.stringify({
              type: "MATCH_FACE",
              isSuccess: true,
              isError: false,
              result: result,
            });
            sendMessage(output);
          } catch (error) {
            let output = JSON.stringify({
              type: "MATCH_FACE",
              isSuccess: false,
              isError: true,
              result: error,
            });
            sendMessage(output);
          }
        } else if (type === "DETECT_FACE") {
          let query = recievedData?.queryImage;
          try {
            const result = await getDetections(query);
            let output = JSON.stringify({
              type: "DETECT_FACE",
              isSuccess: true,
              isError: false,
              result: result,
            });
            sendMessage(output);
          } catch (error) {
            let output = JSON.stringify({
              type: "DETECT_FACE",
              isSuccess: false,
              isError: true,
              result: error,
            });
            sendMessage(output);
          }
        }
      }
    );
    return messageListener;
  }, []);

  const sendMessage = (msg: string) => {
    try {
      // window.postMessage("Open camera called");
      if (window?.ReactNativeWebView?.postMessage) {
        window.ReactNativeWebView.postMessage(msg);
      }
    } catch (error) {}
  };

  const loadModels = useCallback(async () => {
    const MODEL_URL = process.env.PUBLIC_URL + "/models";
    try {
      await Promise.all([
        faceapi.nets.tinyFaceDetector.loadFromUri(MODEL_URL),
        faceapi.nets.faceLandmark68Net.loadFromUri(MODEL_URL),
        faceapi.nets.faceRecognitionNet.loadFromUri(MODEL_URL),
        faceapi.nets.faceExpressionNet.loadFromUri(MODEL_URL),
        faceapi.nets.ssdMobilenetv1.loadFromUri(MODEL_URL),
      ]);
      setModelsLoaded(true);
      /* let output = JSON.stringify({
        type: "MODEL_LOADED",
        isSuccess: true,
        isError: false,
        result: null,
      });
      sendMessage(output); */
    } catch (error) {
      console.log("loadModelsError: ", error);
    }
  }, []);

  useEffect(() => {
    loadModels();
  }, [loadModels]);

  const getDescriptorsForPretrainedImages = async (training: string[]) => {
    const descriptors: {
      name: string;
      descriptor: Float32Array | undefined;
    }[] = [];

    for (let i = 0; i < training.length; i++) {
      const element = training[i];
      const image = await faceapi.fetchImage(element);
      const detections = await faceapi
        .detectSingleFace(image)
        .withFaceLandmarks()
        .withFaceDescriptor();
      descriptors.push({
        name: "Person" + i,
        descriptor: detections?.descriptor,
      });
    }

    return descriptors;
  };

  const checkMatchConfidence = (
    userDescriptors: Float32Array,
    preTrainedDescriptors: {
      name: string;
      descriptor: Float32Array;
    }[]
  ): number => {
    try {
      let distances: number[] = preTrainedDescriptors.map(
        (preTrainedDescriptor) =>
          faceapi.euclideanDistance(
            userDescriptors,
            preTrainedDescriptor.descriptor
          )
      );
      let minDistance = Math.min(...distances);
      console.log({ minDistance });
      minDistance = minDistance * 100;
      let confidence = 100 - minDistance;
      return confidence;
    } catch (error) {
      return 0;
    }
  };

  const matchFaces = async (query: string, training: string[]) => {
    try {
      setResult(undefined);
      setIsMatching(true);
      if (!modelsLoaded) {
        await loadModels();
      }

      const image = await faceapi.fetchImage(query);

      const detections = await faceapi
        .detectSingleFace(image)
        .withFaceLandmarks()
        .withFaceDescriptor();
      if (detections) {
        const pretrainedDescriptors = await getDescriptorsForPretrainedImages(
          training
        );
        const matchResult = checkMatchConfidence(
          detections.descriptor,
          pretrainedDescriptors
        );
        console.log({
          descriptor: detections,
          pretrainedDescriptors,
        });
        setResult(matchResult);
        setIsMatching(false);
        return matchResult;
      } else {
        setResult(0);
        setIsMatching(false);
        return 0;
      }
    } catch (error) {
      sendMessage(error);
    }
  };

  const getDetections = async (arg: string | string[]) => {
    let multiple = false;
    let queryImages: string[] = [];
    if (arg && Array.isArray(arg)) {
      queryImages = arg;
      multiple = true;
    } else {
      queryImages = [arg];
      multiple = false;
    }
    let detections = [];
    for (let i = 0; i < queryImages.length; i++) {
      const query = queryImages[i];
      sendMessage("fetchImage started");
      const image = await faceapi.fetchImage(query);
      sendMessage("fetchImage ended");
      sendMessage("detection started");
      const detection = await faceapi
        .detectSingleFace(image, new faceapi.TinyFaceDetectorOptions())
        .withFaceLandmarks()
        .withFaceDescriptor();
      sendMessage("detection ended");
      if (detection) {
        detections.push(detection);
      }
    }
    if (multiple) {
      return detections;
    } else {
      return detections[0];
    }
  };

  const handleShowCamera = () => {
    dialogRef.current?.showModal();
  };

  const handleHideCamera = () => {
    dialogRef.current?.close();
  };

  const handleUploadTrainingImage = () => {
    handleShowCamera();
    setModalType("TRAINING_IMAGE");
  };
  const handleUploadQueryImage = () => {
    handleShowCamera();
    setModalType("QUERY_IMAGE");
  };

  const handleCapture = () => {
    const imageSrc = webcamRef?.current?.getScreenshot();
    // console.log("imageSrc: ", imageSrc);
    if (modalType === "TRAINING_IMAGE") {
      let _trainingImages = [...trainingImages];
      _trainingImages.push(imageSrc);
      setTrainingImages(_trainingImages);
    } else if (modalType === "QUERY_IMAGE") {
      setQueryImage(imageSrc);
    }
  };

  const prepareFaceDetector = useCallback(() => {
    let base_image = new Image();
    base_image.src = process.env.PUBLIC_URL + "/assets/startFaceDetect.jpg";
    base_image.onload = function () {
      const useTinyModel = true;
      const fullFaceDescription = faceapi
        .detectSingleFace(base_image, new faceapi.TinyFaceDetectorOptions())
        .withFaceLandmarks(useTinyModel)
        .withFaceDescriptor()
        .run()
        .then((res) => {
          let output = JSON.stringify({
            type: "MODEL_LOADED",
            isSuccess: true,
            isError: false,
            result: null,
          });
          sendMessage(output);
        });
    };
  }, []);

  useEffect(() => {
    if (isMatching) {
      loaderDialogRef.current?.showModal();
    } else {
      loaderDialogRef.current?.close();
    }
  }, [isMatching]);

  useEffect(() => {
    if (modelsLoaded) {
      prepareFaceDetector();
    }
  }, [prepareFaceDetector, modelsLoaded]);

  return (
    <div>
      {/* <div className="main-container">
        <div className="card-container">
          <div className="training-card card">
            {trainingImages.map((trainingImage, index) => (
              <div className="img-wrapper" key={index}>
                <img src={trainingImage} height={"100px"} width={"100px"} />
              </div>
            ))}
          </div>
          <div className="card-btn-container">
            <button onClick={handleUploadTrainingImage}>Upload Images</button>
            <button onClick={() => setTrainingImages([])}>Remove All</button>
          </div>
        </div>
        <div className="card-container">
          <div className="query-card card">
            {queryImage ? (
              <div style={{ height: "100%", width: "100%" }}>
                <img src={queryImage} height={"100%"} width={"100%"} />
              </div>
            ) : null}
          </div>
          <div className="card-btn-container">
            <button onClick={handleUploadQueryImage}>Upload Query Image</button>
          </div>
        </div>
      </div>
      <div className="bottom-container  ">
        <button
          onClick={() => {
            if (!isMatching) {
              matchFaces(queryImage, trainingImages);
            }
          }}
          style={{ cursor: isMatching ? "default" : "pointer" }}
        >
          {isMatching ? "Matching..." : "Match"}
        </button>
        {result ? <p>Confidence {Math.round(result)}%</p> : null}
      </div>
      <dialog ref={dialogRef}>
        <div
          style={{
            display: "flex",
            flexDirection: "column",
            borderRadius: "8px",
          }}
        >
          <Webcam ref={webcamRef} />
          <button onClick={handleCapture}>Capture</button>
          <button onClick={handleHideCamera}>Close</button>
        </div>
      </dialog>
      <dialog ref={loaderDialogRef} className="loader-dialog">
        <p>Matching...</p>
      </dialog> */}
    </div>
  );
}

export default App;
