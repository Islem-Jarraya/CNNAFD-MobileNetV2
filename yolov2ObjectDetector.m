%yolov2ObjectDetector Detect objects using YOLO v2 deep learning detector.
%
% Use trainYOLOv2ObjectDetector to train a YOLO v2 object detector. Deep
% Learning Toolbox is required to train and use a YOLO v2 object detector.
%
% detector = ySAVEolov2ObjectDetector(network) create a YOLO v2 object detector
% using a pretrained YOLO v2 network. The input network must be a
% DAGNetwork that has an imageInputLayer and a yolov2TransformLayer
% connected to a yolov2OutputLayer.
%
% [...] = yolov2ObjectDetector(...,Name,Value) specifies additional
% name-value pair arguments described below:
%
% 'TrainingImageSize' - Specify the image sizes used during training as
%                       an M-by-2 matrix, where each row is of the form
%                       [height, width]. The default is the input size
%                       of the network.
%
% yolov2ObjectDetector methods:
%      detect - Detects objects in an image.
%
% yolov2ObjectDetector properties:
%    ModelName          - Name of the trained object detector.
%    Network            - YOLO v2 object detection network. (read-only)
%    ClassNames         - A cell array of object class names. (read-only)
%    AnchorBoxes        - Array of anchor boxes. (read-only)
%    TrainingImageSize  - Array of image sizes used during training.
%                         (read-only)
%
% Example: Detect objects using YOLO v2.
% --------------------------------------
% % Load pre-trained vehicle detector.
% vehicleDetector = load('yolov2VehicleDetector.mat', 'detector');
% detector = vehicleDetector.detector;
%
% % Read test image.
% I = imread('highway.png');
%
% % Run detector.
% [bboxes, scores, labels] = detect(detector, I)
%
% % Display results.
% detectedImg = insertObjectAnnotation(I, 'Rectangle', bboxes, cellstr(labels));
% figure
% imshow(detectedImg)
%
% Example: Train a YOLO v2 object detector.
% -----------------------------------------
% %<a href="matlab:helpview(fullfile(docroot,'toolbox','vision','vision.map'),'ObjectDetectionUsingYOLOV2DeepLearningExample')">Object Detection Using YOLO v2 Deep Learning.</a>
%
% Example: Import a YOLO v2 network.
% ----------------------------------
% %<a href="matlab:helpview(fullfile(docroot,'toolbox','vision','vision.map'),'ImportAPretrainedONNXYOLOV2ObjectDetectorExample')">Import Pretrained ONNX YOLO v2 Object Detector.</a>
%
% Example: Export a YOLO v2 network.
% ----------------------------------
% %<a href="matlab:helpview(fullfile(docroot,'toolbox','vision','vision.map'),'ExportYOLOV2ObjectDetectorToONNXExample')">Export YOLO v2 Object Detector to ONNX.</a>
%
% See also trainYOLOv2ObjectDetector, yolov2Layers, fasterRCNNObjectDetector.

% Copyright 2018-2019 The MathWorks, Inc.

classdef yolov2ObjectDetector < vision.internal.EnforceScalarValue & matlab.mixin.CustomDisplay
    
    properties(GetAccess = public, SetAccess = public)
        % ModelName Name of the trained object detector. By default, the
        % name is set by trainYOLOv2ObjectDetector. The name may be
        % modified after training as desired.
        ModelName   char
    end
    
    properties(SetAccess = protected)
        % Network is a DAGNetwork object representing the YOLO v2 network.
        Network
    end
    
    properties(SetAccess = protected)      
        % An M-by-2 matrix defining the [height width] of image sizes used
        % to train the detector. During detection, an input image is
        % resized to nearest TrainingImageSize before it is processed by
        % the detection network.
        TrainingImageSize
    end
    
    properties(SetAccess = protected, Dependent)
        % AnchorBoxes is an M-by-2 matrix defining the [width height] of M
        % anchor boxes.
        AnchorBoxes        
        
        % ClassNames is a cell array of object class names. These are the
        % object classes that the YOLO v2 detector was trained to find.
        ClassNames
    end        
    
    properties(Access = private, Transient)
        % LayerIndices A struct that caches indices to certain layers used
        %             frequently during detection.
        LayerIndices
    end
    
    properties (Access = protected, Transient)
        FilterBboxesFunctor
    end
    
    properties (Access = public, Hidden)
        % FractionDownsampling can either be true or false. If false, it
        % applies floor operation to the downsampling factor. It is set to
        % true by default.
        FractionDownsampling = true
        
        % WH2HW can either be true or false. If true,
        % nnet.internal.cnn.layer.util.yoloPredictBBox function computes
        % bounding box dimensions using anchor boxes specified in [width,
        % height] format. It is set to false by default.
        WH2HW = false        
    end
    
    methods(Static, Access = public, Hidden)
        %------------------------------------------------------------------
        % Filter Boxes based on size.
        %------------------------------------------------------------------
        function [bboxes, scores, labels] = filterSmallBBoxes(bboxes, scores,labels, minSize)
            [bboxes, scores, labels] = vision.internal.cnn.utils.FilterBboxesFunctor.filterSmallBBoxes(minSize, bboxes, scores, labels);
        end
        
        function [bboxes, scores, labels] = filterLargeBBoxes(bboxes, scores, labels, maxSize)
            [bboxes, scores, labels] = vision.internal.cnn.utils.FilterBboxesFunctor.filterLargeBBoxes(maxSize, bboxes, scores, labels);
        end
        
        %------------------------------------------------------------------
        % Detector checkpoint function for yolov2.
        %------------------------------------------------------------------
        function detector = detectorCheckpoint(net, detector)
            detector.Network = net;
        end
        
        %------------------------------------------------------------------
        function A = preprocess(A,outputSize)
            % Rescale and cast data to single.
            if iscell(A)
                [A{1:min(2,numel(A))}] = iPreprocess(outputSize,A{:});
            else
                A = iPreprocess(outputSize,A);
            end
        end
        
        %------------------------------------------------------------------
        function A = trainingTransformForDatastore(A,outputSize)
            A = yolov2ObjectDetector.preprocess(A,outputSize);
            A = iAppendResponsesForTraining(A,outputSize);
        end
        
        %------------------------------------------------------------------
        function A = trainingTransform(A,outputSize)
            A = iAugmentData(A);
            A = yolov2ObjectDetector.preprocess(A,outputSize);
            A = iAppendResponsesForTraining(A,outputSize);
        end
        
        %------------------------------------------------------------------
        function A = validationTransform(A,outputSize)
            A = yolov2ObjectDetector.preprocess(A,outputSize);
            A = iAppendResponsesForTraining(A,outputSize);
        end
        
        %------------------------------------------------------------------
        function printHeader(printer, classNames)
            printer.print('*************************************************************************\n');
            printer.printMessage('vision:yolo:trainingBanner');
            printer.linebreak;
            
            for i = 1:numel(classNames)
                printer.print('* %s\n', classNames{i});
            end
            
            printer.linebreak;
        end
        
        %------------------------------------------------------------------
        function printFooter(printer)
            printer.printMessage('vision:yolo:trainingFooter');
            printer.print('*************************************************************************\n');
            printer.linebreak;
        end
        
    end
    %----------------------------------------------------------------------
    methods
        function varargout = detect(this, I, varargin)
            % bboxes = detect(yolo,I) detects objects within the image I.
            % The location of objects within I are returned in bboxes, an
            % M-by-4 matrix defining M bounding boxes. Each row of bboxes
            % contains a four-element vector, [x, y, width, height]. This
            % vector specifies the upper-left corner and size of a bounding
            % box in pixels. yolo is a yolov2ObjectDetector object
            % and I is a truecolor or grayscale image.
            %
            % bboxes = detect(yolo,IBatch) detects objects within each
            % image contained in the batch of images IBatch. IBatch is a
            % numeric array containing images in the format
            % H-by-W-by-C-by-B, where B is the number of images in the
            % batch, and C is the channel size. For grayscale images, C must be
            % 1. The network input channel size of the detector, yolo, must
            % match the channel size of each image in the batch, H-by-W-by-C-by-B.
            % bboxes is a B-by-1 cell array, containing M-by-4 matrices for
            % each image in the batch.
            %
            % [..., scores] = detect(yolo,I) optionally return the class
            % specific confidence scores for each bounding box. The scores
            % for each detection is product of objectness prediction and
            % classification scores. The range of the scores is [0 1].
            % Larger score values indicate higher confidence in the
            % detection. scores is a B-by-1 cell array, if the input I is
            % a batch of images in the format H-by-W-by-C-by-B.
            %
            % [..., labels] = detect(yolo,I) optionally return the labels
            % assigned to the bounding boxes in an M-by-1 categorical
            % array. The labels used for object classes is defined during
            % training using the trainYOLOv2ObjectDetector function.
            % labels is a B-by-1 cell array, if the input I is a batch of
            % images in the format H-by-W-by-C-by-B.
            %
            % detectionResults = detect(yolo,DS) detects objects within the
            % series of images returned by the read method of datastore,
            % DS. DS, must be a datastore that returns a table or a cell
            % array with the first column containing images.
            % detectionResults is a 3-column table with variable names
            % 'Boxes', 'Scores', and 'Labels' containing bounding boxes,
            % scores, and the labels. The location of objects within an
            % image, I are returned in bounding boxes, an M-by-4 matrix
            % defining M bounding boxes. Each row of boxes contains a
            % four-element vector, [x, y, width, height]. This vector
            % specifies the upper-left corner and size of a bounding box in
            % pixels. yolo is a yolov2ObjectDetector object.
            %
            % [...] = detect(..., roi) optionally detects objects within
            % the rectangular search region specified by roi. roi must be a
            % 4-element vector, [x, y, width, height], that defines a
            % rectangular region of interest fully contained in I.
            %
            % [...] = detect(..., Name, Value) specifies additional
            % name-value pairs described below:
            %
            % 'Threshold'              A scalar between 0 and 1. Detections
            %                          with scores less than the threshold
            %                          value are removed. Increase this value
            %                          to reduce false positives.
            %
            %                          Default: 0.5
            %
            % 'SelectStrongest'        A logical scalar. Set this to true to
            %                          eliminate overlapping bounding boxes
            %                          based on their scores. This process is
            %                          often referred to as non-maximum
            %                          suppression. Set this to false if you
            %                          want to perform a custom selection
            %                          operation. When set to false, all the
            %                          detected bounding boxes are returned.
            %
            %                          Default: true
            %
            % 'MinSize'                Specify the size of the smallest
            %                          region containing an object, in
            %                          pixels, as a two-element vector,
            %                          [height width]. When the minimum size
            %                          is known, you can reduce computation
            %                          time by setting this parameter to that
            %                          value. By default, 'MinSize' is the
            %                          smallest object that can be detected
            %                          by the trained network.
            %
            %                          Default: [1,1]
            %
            % 'MaxSize'                Specify the size of the biggest region
            %                          containing an object, in pixels, as a
            %                          two-element vector, [height width].
            %                          When the maximum object size is known,
            %                          you can reduce computation time by
            %                          setting this parameter to that value.
            %                          Otherwise, the maximum size is
            %                          determined based on the width and
            %                          height of I.
            %
            %                          Default: size(I)
            %
            % 'MiniBatchSize'          The mini-batch size used for processing a
            %                          large collection of images. Images are grouped
            %                          into mini-batches and processed as a batch to
            %                          improve computational efficiency. Larger
            %                          mini-batch sizes lead to faster processing, at
            %                          the cost of more memory.
            %
            %                          Default: 128
            %
            % 'ExecutionEnvironment'   The hardware resources used to run the
            %                          YOLO v2 detector. Valid values are:
            %
            %                          'auto' - Use a GPU if it is available,
            %                                   otherwise use the CPU.
            %
            %                           'gpu' - Use the GPU. To use a GPU,
            %                                   you must have Parallel
            %                                   Computing Toolbox(TM), and a
            %                                   CUDA-enabled NVIDIA GPU with
            %                                   compute capability 3.0 or
            %                                   higher. If a suitable GPU is
            %                                   not available, an error
            %                                   message is issued.
            %
            %                           'cpu  - Use the CPU.
            %
            %                          Default : 'auto'
            %
            % 'Acceleration'           Optimizations that can improve
            %                          performance at the expense of some
            %                          overhead on the first call, and possible
            %                          additional memory usage. Valid values
            %                          are:
            %
            %                           'auto'    - Automatically select
            %                                       optimizations suitable
            %                                       for the input network and
            %                                       environment.
            %
            %                           'mex'     - (GPU Only) Generate and
            %                                       execute a MEX function.
            %
            %                           'none'    - Disable all acceleration.
            %
            %                          Default : 'auto'
            %
            %  Notes:
            %  -----
            %  - When 'SelectStrongest' is true the selectStrongestBboxMulticlass
            %    function is used to eliminate overlapping boxes. By
            %    default, the function is called as follows:
            %
            %   selectStrongestBboxMulticlass(bbox,scores,labels,...
            %                                       'RatioType', 'Union', ...
            %                                       'OverlapThreshold', 0.5);
            %
            %  - When the input image size does not match the network input size, the
            %    detector resizes the input image to the network input size.
            %
            % Class Support
            % -------------
            % The input image I can be uint8, uint16, int16, double,
            % single, or logical, and it must be real and non-sparse.
            %
            % Example
            % -------
            % % Load pre-trained vehicle detector.
            % vehicleDetector = load('yolov2VehicleDetector.mat', 'detector');
            % detector = vehicleDetector.detector;
            %
            % % Read test image.
            % I = imread('highway.png');
            %
            % % Run detector.
            % [bboxes, scores, labels] = detect(detector, I)
            %
            % % Display results.
            % detectedImg = insertObjectAnnotation(I, 'Rectangle', bboxes, cellstr(labels));
            % figure
            % imshow(detectedImg)
            %
            % See also trainYOLOv2ObjectDetector, selectStrongestBboxMulticlass.
            
            params = this.parseDetectInputs(I,varargin{:});
            
            params.ClassNames = this.ClassNames;
            
            roi    = params.ROI;
            useROI = params.UseROI;
            
            anchors = this.AnchorBoxes;
            trainingImageSize = this.TrainingImageSize;
            
            layerName = this.Network.Layers(this.LayerIndices.OutputLayerIdx).Name;
            params.FilterBboxesFunctor = this.FilterBboxesFunctor;
            params.FractionDownsampling = this.FractionDownsampling;
            params.WH2HW = this.WH2HW;
            if params.DetectionInputWasDatastore
                nargoutchk(0,1);
                
                % Copy and reset the given datastore, so external state events are
                % not reflected.
                ds = copy(I);
                reset(ds);
                
                fcn = @iPreprocessForDetect;
                % We need just the preprocessed image -> num arg out is 1.
                fcnArgOut = 2;
                ds = transform(ds, @(x)iPreProcessForDatastoreRead(x,fcn,fcnArgOut,roi,useROI,trainingImageSize));
                % Process datastore with network and output the predictions.
                params.TrainingImageSize = trainingImageSize;
                varargout{1} = iPredictUsingDatastore(ds, this.Network, params, anchors, layerName);
            else
                nargoutchk(0,3);
                
                [Ipreprocessed,info]= iPreprocessForDetect(I,roi,useROI,trainingImageSize);
                
                features = iGetFeaturesUsingActivations(this.Network,Ipreprocessed,layerName,params);

                if params.DetectionInputWasBatchOfImages
                    [varargout{1:nargout}] = iPostProcessBatchActivations(features, info, anchors, params);
                else
                    [varargout{1:nargout}] = iPostProcessActivations(features, info, anchors, params);
                end
            end
        end
    end
    %----------------------------------------------------------------------
    methods
        function this = yolov2ObjectDetector(varargin)
            narginchk(1,3);
            if (isa(varargin{1,1},'yolov2ObjectDetector'))
                clsname = 'yolov2ObjectDetector';
                validateattributes(varargin{1,1},{clsname}, ...
                    {'scalar'}, mfilename);
                this.ModelName            = varargin{1,1}.ModelName;
                this.Network              = varargin{1,1}.Network;
                this.TrainingImageSize    = varargin{1,1}.TrainingImageSize;
                this.FractionDownsampling = varargin{1,1}.FractionDownsampling;
                this.WH2HW                = varargin{1,1}.WH2HW;
            else
                % Configure detector.
                this.ModelName   = 'importedNetwork';
                params = this.parseDetectorInputs(varargin{:});
                this.TrainingImageSize = params.TrainingImageSize;
                this.Network = params.Network;
            end
            this.FilterBboxesFunctor = vision.internal.cnn.utils.FilterBboxesFunctor;
        end
    end
    %----------------------------------------------------------------------
    methods
        function this = set.Network(this, network)
            validateattributes(network, ...
                {'DAGNetwork'},{'scalar'});
            
            % update layer index cache
            this = this.setLayerIndices(network);
            this.Network = network;
        end
        
        %------------------------------------------------------------------
        function anchorBoxes = get.AnchorBoxes(this)
            outputLayerIdx = this.LayerIndices.OutputLayerIdx;
            anchorBoxes = this.Network.Layers(outputLayerIdx,1).AnchorBoxes;
        end         
        
        %------------------------------------------------------------------
        function classes = get.ClassNames(this)
            outputLayerIdx = this.LayerIndices.OutputLayerIdx;
            classesTmp = cellstr(this.Network.Layers(outputLayerIdx,1).Classes);
            if ~iscategorical(classesTmp)
                classesTmp = categorical(classesTmp);
            end
            classes = classesTmp';
        end        
    end
    %----------------------------------------------------------------------
    methods (Static)
        % Create yolov2Datastore for training.
        function ds = createYoloTrainingDatastore(trainingData,dsOpts)
            ds = vision.internal.cnn.yolo.yolov2Datastore(trainingData,dsOpts);
        end
        
        %------------------------------------------------------------------
        function mapping = createMIMODatastoreMapping(ds, lgraph, params)
            externalLayers = lgraph.Layers;
            
            % Get layer name based on location in Layers array.
            lossName = externalLayers(params.yoloOutputLayerIdx).Name;
            imgName = externalLayers(params.inputImageIdx).Name;
            
            dst = {
                imgName
                lossName
                };
            
            mapping = table(ds.OutputTableVariableNames',dst,...
                'VariableNames',{'Data','Destination'});
        end
        
        %------------------------------------------------------------------
        function mapping = createMIMODatastoreCellMapping(inputLayerSize)
            % There is one input layer: ImageInputLayer.
            %  - the first column from the read output goes to the input layer.
            inputMapping = {1};
            % There is one output layer: YOLOv2OutputLayer.
            %    - the second column contains responses for this layer.
            outputMapping = {2};
            
            % Output layer is not a classification layer, but a regression layer.
            classificationOutputs = false;
            
            inputSizes = {inputLayerSize};
            % We make the outputLayerSize empty, so the dispatcher considers the output observation dimension as 1.
            outputSizes = {[]};
            mapping = {inputMapping, outputMapping, classificationOutputs, inputSizes, outputSizes};
        end
    end
    %----------------------------------------------------------------------
    methods (Access = protected)
        %------------------------------------------------------------------
        % Parse and validate detection parameters.
        %------------------------------------------------------------------
        function params = parseDetectInputs(this, I, varargin)
            
            params.DetectionInputWasDatastore = ~isnumeric(I);
            
            if params.DetectionInputWasDatastore
                sampleImage = vision.internal.cnn.validation.checkDetectionInputDatastore(I, mfilename);
            else
                sampleImage = I;
            end

            network = this.Network;

            networkInputSize = network.Layers(this.LayerIndices.ImageLayerIdx).InputSize;

            validateChannelSize = true;  % check if the channel size is equal to that of the network 
            validateImageSize   = false; % yolov2 can support images smaller than input size
            [sz,params.DetectionInputWasBatchOfImages] = vision.internal.cnn.validation.checkDetectionInputImage(...
                networkInputSize,sampleImage,validateChannelSize,validateImageSize);
            
            defaults = iDefaultDetectionParams();
            
            p = inputParser;
            p.addOptional('roi', defaults.roi);
            p.addParameter('SelectStrongest', defaults.SelectStrongest);
            p.addParameter('MinSize', defaults.MinSize);
            p.addParameter('MaxSize', sz(1:2));
            p.addParameter('MiniBatchSize', defaults.MiniBatchSize);
            p.addParameter('ExecutionEnvironment', defaults.ExecutionEnvironment);
            p.addParameter('Acceleration', 'auto');
            p.addParameter('Threshold', defaults.Threshold);
            parse(p, varargin{:});
            
            userInput = p.Results;

            vision.internal.cnn.validation.checkMiniBatchSize(userInput.MiniBatchSize, mfilename);
            
            
            useROI = ~ismember('roi', p.UsingDefaults);
            
            if useROI
                vision.internal.detector.checkROI(userInput.roi, sz);
            end
            
            vision.internal.inputValidation.validateLogical(...
                userInput.SelectStrongest, 'SelectStrongest');
            
            % Validate minsize and maxsize.
            validateMinSize = ~ismember('MinSize', p.UsingDefaults);
            validateMaxSize = ~ismember('MaxSize', p.UsingDefaults);
            
            if validateMinSize
                vision.internal.detector.ValidationUtils.checkMinSize(userInput.MinSize, [1,1], mfilename);
            end
            
            if validateMaxSize
                vision.internal.detector.ValidationUtils.checkSize(userInput.MaxSize, 'MaxSize', mfilename);
                if useROI
                    coder.internal.errorIf(any(userInput.MaxSize > userInput.roi([4 3])) , ...
                        'vision:yolo:modelMaxSizeGTROISize',...
                        userInput.roi(1,4),userInput.roi(1,3));
                else
                    coder.internal.errorIf(any(userInput.MaxSize > sz(1:2)) , ...
                        'vision:yolo:modelMaxSizeGTImgSize',...
                        sz(1,1),sz(1,2));
                end
            end
            
            if validateMaxSize && validateMinSize
                coder.internal.errorIf(any(userInput.MinSize >= userInput.MaxSize) , ...
                    'vision:ObjectDetector:minSizeGTMaxSize');
            end
            
            if useROI
                if ~isempty(userInput.roi)
                    sz = userInput.roi([4 3]);
                    vision.internal.detector.ValidationUtils.checkImageSizes(sz(1:2), userInput, validateMinSize, ...
                        userInput.MinSize, ...
                        'vision:ObjectDetector:ROILessThanMinSize', ...
                        'vision:ObjectDetector:ROILessThanMinSize');
                end
            else
                vision.internal.detector.ValidationUtils.checkImageSizes(sz(1:2), userInput, validateMaxSize, ...
                    userInput.MinSize , ...
                    'vision:ObjectDetector:ImageLessThanMinSize', ...
                    'vision:ObjectDetector:ImageLessThanMinSize');
            end
            
            % Validate threshold.
            yolov2ObjectDetector.checkThreshold(userInput.Threshold);
            
            % Validate execution environment.
            exeEnv = vision.internal.cnn.validation.checkExecutionEnvironment(...
                userInput.ExecutionEnvironment, mfilename);
            
            accel = vision.internal.cnn.validation.checkAcceleration(...
                userInput.Acceleration, mfilename);
            
            params.ROI                      = single(userInput.roi);
            params.UseROI                   = useROI;
            params.SelectStrongest          = logical(userInput.SelectStrongest);
            params.MinSize                  = single(userInput.MinSize);
            params.MaxSize                  = single(userInput.MaxSize);
            params.MiniBatchSize            = double(userInput.MiniBatchSize);
            params.Threshold                = single(userInput.Threshold);
            params.ExecutionEnvironment     = exeEnv;
            params.Acceleration             = accel;
            params.NetworkInputSize         = networkInputSize;
        end
        
        %------------------------------------------------------------------
        function this = setLayerIndices(this, network)
            this.LayerIndices.OutputLayerIdx = yolov2ObjectDetector.findYOLOv2OutputLayer(network.Layers);
            this.LayerIndices.ImageLayerIdx = yolov2ObjectDetector.findYOLOv2ImageInputLayer(network.Layers);
        end
        
        %------------------------------------------------------------------
        % Parse and validate detector parameters.
        %------------------------------------------------------------------
        function params = parseDetectorInputs(~,varargin)
            network = iValidateNetwork(varargin{1,1});
            
            % Extract class names, anchor boxes.
            outputLayerIdx = yolov2ObjectDetector.findYOLOv2OutputLayer(network.Layers);
            
            if isempty(outputLayerIdx)
                error(message("vision:yolo:mustHaveOutputLayer"));
            else
                classNames = network.Layers(outputLayerIdx,1).Classes;
                if ischar(classNames)
                    error(message("vision:yolo:mustHaveClassNames"));
                end
                anchorBoxes = network.Layers(outputLayerIdx,1).AnchorBoxes;
            end
            
            numClasses = size(classNames,1);
            yoloLgraph = layerGraph(network);
            iValidateNetworkLayers(numClasses,yoloLgraph)
            
            p = inputParser;
            p.addRequired('Network')
            
            inputIdx = yolov2ObjectDetector.findYOLOv2ImageInputLayer(network.Layers);
            networkImageSize = network.Layers(inputIdx,1).InputSize(1,1:2);
            p.addParameter('TrainingImageSize', networkImageSize);
            
            parse(p, varargin{:});
            params.Network = p.Results.Network;
            params.TrainingImageSize = p.Results.TrainingImageSize;
            iCheckTrainingImageSize(params.TrainingImageSize,networkImageSize);
            
            params.ClassNames = classNames';
            params.AnchorBoxes = anchorBoxes;
        end
    end
    
    %======================================================================
    % Save/Load
    %======================================================================
    methods(Hidden)
        function s = saveobj(this)
            s.Version                  = 4.0;
            s.ModelName                = this.ModelName;
            s.Network                  = this.Network;
            s.ClassNames               = this.ClassNames;
            s.AnchorBoxes              = this.AnchorBoxes;
            s.TrainingImageSize        = this.TrainingImageSize;
            s.FractionDownsampling     = this.FractionDownsampling;
            s.WH2HW                    = this.WH2HW;
        end 
        
    end
    
    methods(Static, Hidden)
        function this = loadobj(s)
            try
                vision.internal.requiresNeuralToolbox(mfilename);
                switch s.Version
                    case 1 % <= 19b
                        s = iUpgradeToVersionFour(s);
                        s.FractionDownsampling = false;
                        s.WH2HW = true;
                    case 2 % == 19b update 1
                        s = iUpgradeToVersionFour(s);
                        s.WH2HW = true;
                    case 3
                        s = iUpgradeToVersionFour(s);
                        s.WH2HW = true;
                    otherwise
                        % no-op.
                end
                trainingImgSize = s.TrainingImageSize;
                this = yolov2ObjectDetector(s.Network,'TrainingImageSize',trainingImgSize);
                this.ModelName                = s.ModelName;
                this.FractionDownsampling     = s.FractionDownsampling;
                this.WH2HW                    = s.WH2HW;
            catch ME
                rethrow(ME)
            end
        end
    end
    %----------------------------------------------------------------------
    % Assemble detector object for training.
    %----------------------------------------------------------------------
    methods(Hidden, Static)
        function detector = assembleDetector(params,net)
            trainingImgSize = params.TrainingImageSize;
            detector = yolov2ObjectDetector(net,'TrainingImageSize',trainingImgSize);
            detector.ModelName   = params.ModelName;
        end
        
        %------------------------------------------------------------------
        % Update class names of yolov2OutputLayer in network.
        %------------------------------------------------------------------
        function updatedLgraph = updateNetworkClasses(lgraph, classNames)
            % yolov2OutputLayer created without class names. In such cases
            % the classes are obtained from input.
            outputLayerIdx = ...
                arrayfun(@(x)isa(x,'nnet.cnn.layer.YOLOv2OutputLayer'), ...
                lgraph.Layers);        
            outputLayer = lgraph.Layers(outputLayerIdx,1);
            outputLayer.Classes = classNames;
            updatedLgraph = replaceLayer(lgraph,outputLayer.Name,outputLayer);
        end
        %------------------------------------------------------------------
        % Validate Threshold value.
        %------------------------------------------------------------------
        function checkThreshold(threshold)
            validateattributes(threshold, {'single', 'double'}, {'nonempty', 'nonnan', ...
                'finite', 'nonsparse', 'real', 'scalar', '>=', 0, '<=', 1}, ...
                mfilename, 'Threshold');
        end
        
        %------------------------------------------------------------------
        % Find indexes of yolov2OutputLayer and ImageInputLayer in network.
        %------------------------------------------------------------------
        function yoloOutputLayerIdx = findYOLOv2OutputLayer(externalLayers)
            yoloOutputLayerIdx = find(...
                arrayfun( @(x)isa(x,'nnet.cnn.layer.YOLOv2OutputLayer'), ...
                externalLayers));
        end
        
        function imageInputIdx = findYOLOv2ImageInputLayer(externalLayers)
            imageInputIdx = find(...
                arrayfun( @(x)isa(x,'nnet.cnn.layer.ImageInputLayer'), ...
                externalLayers));
        end
        
    end
end

%--------------------------------------------------------------------------
function s = iDefaultDetectionParams()
s.roi                     = zeros(0,4);
s.SelectStrongest         = true;
s.Threshold               = 0.5;
s.MinSize                 = [1,1];
s.MaxSize                 = [];
s.MiniBatchSize           = 128;
s.ExecutionEnvironment    = 'auto';
end

%--------------------------------------------------------------------------
function network = iValidateNetwork(network)
validateattributes(network,{'DAGNetwork'}, ...
    {'scalar'}, mfilename);
end

%--------------------------------------------------------------------------
function S = iUpgradeToVersionFour(S)
% iUpgradeToVersionThree   Upgrade a v1 or v2 or v3 saved struct to a v4
% saved struct.
%   This means adding class names to yolov2OutputLayer.
S.Version = 4;
lgraph = layerGraph(S.Network);
updatedLgraph = yolov2ObjectDetector.updateNetworkClasses(lgraph, S.ClassNames);
updatedNetwork = assembleNetwork(updatedLgraph);
S.Network = updatedNetwork;
end

%--------------------------------------------------------------------------
function iValidateNetworkLayers(numClasses,yolov2LayerGraph)
analysis = nnet.internal.cnn.analyzer.NetworkAnalyzer(yolov2LayerGraph);
constraint = vision.internal.cnn.analyzer.constraints.YOLOv2Architecture(numClasses);
out = nnet.internal.cnn.analyzer.constraints.Constraint.getBuiltInConstraints();
archConstraint = arrayfun(@(x)isa(x,'nnet.internal.cnn.analyzer.constraints.Architecture'),out);
out(archConstraint) = constraint;
analysis.applyConstraints(out);
try
    analysis.throwIssuesIfAny();
catch ME
    throwAsCaller(ME);
end
end

%--------------------------------------------------------------------------
function iCheckTrainingImageSize(trainingImageSize,networkImageSize)
validateattributes(trainingImageSize, {'numeric'}, ...
    {'2d','ncols',2,'ndims',2,'nonempty','nonsparse',...
    'real','finite','integer','positive'});
if any(trainingImageSize < networkImageSize,'all')
    error(message('vision:yolo:multiScaleInputError'))
end
end

%--------------------------------------------------------------------------
function clippedBBox = iClipBBox(bbox, imgSize)

clippedBBox  = double(bbox);

x1 = clippedBBox(:,1);
y1 = clippedBBox(:,2);

x2 = clippedBBox(:,3);
y2 = clippedBBox(:,4);

x1(x1 < 1) = 1;
y1(y1 < 1) = 1;

x2(x2 > imgSize(2)) = imgSize(2);
y2(y2 > imgSize(1)) = imgSize(1);

clippedBBox = [x1 y1 x2 y2];
end

%--------------------------------------------------------------------------
function [targetSize, sx, sy] = iFindNearestTrainingImageSize(sz,trainingImageSize)
idx = iComputeBestMatch(sz(1:2),trainingImageSize);
targetSize = trainingImageSize(idx,:);

% Compute scale factors to scale boxes from targetSize back to the input
% size.
scale   = sz(1:2)./targetSize;
[sx,sy] = deal(scale(2),scale(1));
end

%--------------------------------------------------------------------------
% Get the index of nearest size in TrainingImageSize training sizes that
% matches given image.
%--------------------------------------------------------------------------
function ind = iComputeBestMatch(preprocessedImageSize,trainingImageSize)
preprocessedImageSize = repmat(preprocessedImageSize,size(trainingImageSize,1),1);
Xdist = (preprocessedImageSize(:,1) - trainingImageSize(:,1));
Ydist = (preprocessedImageSize(:,2) - trainingImageSize(:,2));
dist = sqrt(Xdist.^2 + Ydist.^2);
[~,ind] = min(dist);
end

%--------------------------------------------------------------------------
% Process image with network and outputPredictions in the following format:
% outputPredictions(:, 1:4)   - output boxes in [x1 y1 x2 y2] format.
% outputPredictions(:, 5)     - scores in M-by-1 vector format.
% outputPredictions(:, 6:end) - labels in M-by-1 vector format.
%--------------------------------------------------------------------------
function outputPrediction = iPredictUsingFeatureMap(featureMap, threshold, preprocessedImageSize, anchorBoxes, fractionDownsampling, wh2hw)

gridSize = size(featureMap);

featureMap = permute(featureMap,[2 1 3 4]);
featureMap = reshape(featureMap,gridSize(1)*gridSize(2),gridSize(3),1,[]);
featureMap = reshape(featureMap,gridSize(1)*gridSize(2),size(anchorBoxes,1),gridSize(3)/size(anchorBoxes,1),[]);
featureMap = permute(featureMap,[2 3 1 4]);

% This is to maintain backward compatibility with version 1 detectors.
if fractionDownsampling
    downsampleFactor = preprocessedImageSize(1:2)./gridSize(1:2);
else
    downsampleFactor = floor(preprocessedImageSize(1:2)./gridSize(1:2));
end

if wh2hw
   anchorBoxes = [anchorBoxes(:,2),anchorBoxes(:,1)];
end

% Scale anchor boxes with respect to feature map size
anchorBoxes = anchorBoxes./downsampleFactor;

% Extract IoU, class probabilities from feature map.
iouPred = featureMap(:,1,:,:);
sigmaXY = featureMap(:,2:3,:,:);
expWH = featureMap(:,4:5,:,:);
probPred = featureMap(:,6:end,:,:);

% Compute bounding box coordinates [x,y,w,h] with respect to input image
% dimension.
boxOut = nnet.internal.cnn.layer.util.yoloPredictBBox(sigmaXY, expWH, anchorBoxes, gridSize(1:2), downsampleFactor);

boxOut = permute([boxOut,iouPred,probPred],[2 1 3 4]);
boxOut = reshape(boxOut,size(boxOut,1),[]);
boxOut = permute(boxOut,[2 1 3 4]);

% Extract box coordinates, iou, class probabilities.
bboxesX1Y1X2Y2 = boxOut(:,1:4);
iouPred = boxOut(:,5);
probPred = boxOut(:,6:end);
[imax,idx] = max(probPred,[],2);
confScore = iouPred.*imax;
boxOut = [bboxesX1Y1X2Y2,confScore,idx];
save C:\Users\Jarraya\Desktop\islem\horse\boxOut boxOut
outputPrediction = boxOut(confScore>=threshold,:);
end

%--------------------------------------------------------------------------
function detectionResults = iPredictUsingDatastore(ds, network, params, anchorBoxes, layerName)

loader = iCreateDataLoader(ds,params.MiniBatchSize,params.NetworkInputSize);

% Iterate through data and write results to disk.
k = 1;

bboxes = cell(params.MiniBatchSize, 1);
scores = cell(params.MiniBatchSize, 1);
labels = cell(params.MiniBatchSize, 1);

while hasdata(loader)
    X = nextBatch(loader);
    imgBatch = X{1};
    batchInfo = X{2};
    numMiniBatch = size(batchInfo,1);
    features = iGetFeaturesUsingActivations(network,imgBatch,layerName,params);
    for ii = 1:numMiniBatch
        fmap = features(:,:,:,ii);
        [bboxes{k},scores{k},labels{k}] = ...
            iPostProcessActivations(fmap, batchInfo{ii}, anchorBoxes, params);
        k = k + 1;
    end
end

varNames = {'Boxes', 'Scores', 'Labels'};
detectionResults = table(bboxes(1:k-1), scores(1:k-1), labels(1:k-1), 'VariableNames', varNames);
end

%-----------------------------------------------------------------------
function features = iGetFeaturesUsingActivations(network,imgBatch,layerName,params)
try
    [h,w,c,~] = size(imgBatch);
    canUsePredict = isequal([h w c],params.NetworkInputSize);
    if canUsePredict
        % Use predict when the image batch size matches the network input
        % size as it has faster inference compared to activations.
        features = predict(network,imgBatch,...
            'Acceleration',params.Acceleration,...
            'ExecutionEnvironment',params.ExecutionEnvironment);
    else
        features = activations(network,imgBatch,layerName,...
            'Acceleration',params.Acceleration,...
            'ExecutionEnvironment',params.ExecutionEnvironment);
    end
catch ME
    if strcmp(ME.identifier,'nnet_cnn:layer:BatchNormalizationLayer:NotFinalized')
        error(message('vision:yolo:unableToDetect'));
    else
        throwAsCaller(ME);
    end
end
end

%-----------------------------------------------------------------------
function [bboxes, scores, labels] = iPostProcessBatchActivations(features, info, anchorBoxes, params)
    numMiniBatch = size(features,4);
    bboxes = cell(numMiniBatch, 1);
    scores = cell(numMiniBatch, 1);
    labels = cell(numMiniBatch, 1);

    for ii = 1:numMiniBatch
        fmap = features(:,:,:,ii);
        [bboxes{ii},scores{ii},labels{ii}] = ...
            iPostProcessActivations(fmap, info, anchorBoxes, params);
    end
end

%-----------------------------------------------------------------------
function [bboxes, scores, labels] = iPostProcessActivations(featureMap, info, anchorBoxes, params)

outputPrediction = iPredictUsingFeatureMap(featureMap, params.Threshold, info.PreprocessedImageSize, anchorBoxes, params.FractionDownsampling, params.WH2HW);

if ~isempty(outputPrediction)
    
    bboxesX1Y1X2Y2 = outputPrediction(:,1:4);
    scorePred = outputPrediction(:,5);
    classPred = outputPrediction(:,6);
    
    % ClipBoxes to boundaries.
    bboxesX1Y1X2Y2 = iClipBBox(bboxesX1Y1X2Y2, info.PreprocessedImageSize);
    
    % Scale boxes back to size(Iroi).
    bboxesX1Y1X2Y2 = vision.internal.cnn.boxUtils.scaleX1X2Y1Y2(bboxesX1Y1X2Y2, info.ScaleX, info.ScaleY);
    
    % Convert [x1 y1 x2 y2] to [x y w h].
    bboxPred = vision.internal.cnn.boxUtils.x1y1x2y2ToXYWH(bboxesX1Y1X2Y2);
    
    % Filter boxes based on MinSize, MaxSize.
    [bboxPred, scorePred, classPred] = filterBBoxes(params.FilterBboxesFunctor,...
        params.MinSize,params.MaxSize,bboxPred,scorePred,classPred);
    
    % Apply NMS.
    if params.SelectStrongest
        [bboxes, scores, labels] = selectStrongestBboxMulticlass(bboxPred, scorePred, classPred ,...
            'RatioType', 'Union', 'OverlapThreshold', 0.5);
    else
        bboxes = bboxPred;
        scores = scorePred;
        labels = classPred;
    end
    
    % Apply ROI offset
    bboxes(:,1:2) = vision.internal.detector.addOffsetForROI(bboxes(:,1:2), params.ROI, params.UseROI);
    
    % Convert classId to classNames.
    classnames = params.ClassNames;
    labels = classnames(1,labels);
    labels = categorical(cellstr(labels))';
    
else
    bboxes = zeros(0,4,'single');
    scores = zeros(0,1,'single');
    labels = categorical(cell(0,1),cellstr(params.ClassNames));
end

end

%--------------------------------------------------------------------------
function B = iAugmentData(A)
% Apply random horizontal flipping, and random X/Y scaling. Boxes that get
% scaled outside the bounds are clipped if the overlap is above 0.25. Also,
% jitter image color.
B = cell(size(A));

I = A{1};
sz = size(I);
if numel(sz)==3 && sz(3) == 3
    I = jitterColorHSV(I,...
        'Contrast',0.2,...
        'Hue',0,...
        'Saturation',0.1,...
        'Brightness',0.2);
end

% Randomly flip and scale image.
tform = randomAffine2d('XReflection',true,'Scale',[1 1.1]);
rout = affineOutputView(sz,tform,'BoundsStyle','CenterOutput');
B{1} = imwarp(I,tform,'OutputView',rout);

% Apply same transform to boxes.
[B{2},indices] = bboxwarp(A{2},tform,rout,'OverlapThreshold',0.25);
B{3} = A{3}(indices);

% Return original data only when all boxes are removed by warping.
if isempty(indices)
    B = A;
end
end

%--------------------------------------------------------------------------
function A = iAppendResponsesForTraining(A,outputSize)

% Append label IDs to each box.
A{2} = [A{2} double(A{3})];

% Remove undefined boxes.
A{2}(ismissing(A{3}),:) = [];

% Append output size to each box.
N = size(A{2},1);
A{2} = [A{2} repelem(outputSize(1:2),N,1)];

% Delete labels
A(3) = [];
end

%--------------------------------------------------------------------------
function [I,varargout] = iPreprocess(outputSize,I,varargin)
% Resize image and boxes, then normalize image data between 0 and 1.
sz = size(I);
I = imresize(I,outputSize(1:2));
I = vision.internal.cnn.yolo.yolov2Datastore.normalizeImageAndCastToSingle(I);
if numel(varargin) > 1
    % Resize boxes using the same scale factor as the image resize.
    scale = outputSize(1:2)./sz(1:2);
    varargout{1} = bboxresize(varargin{1},scale);
end
end

%--------------------------------------------------------------------------
function out = iPreProcessForDatastoreRead(in, fcn, numArgOut, varargin)
if isnumeric(in)
    % Numeric input
    in = {in};
end
if istable(in)
    % Table input
    in = in{:,1};
else
    % Cell input
    in = in(:,1);
end
numItems = numel(in);
out = cell(numItems, numArgOut);
for ii = 1:numel(in)
    [out{ii, 1:numArgOut}] = fcn(in{ii},varargin{:});
end
end

%--------------------------------------------------------------------------
function [Ipreprocessed,info] = iPreprocessForDetect(I, roi, useROI, trainingImageSize)
% Crop image if requested.
Iroi = vision.internal.detector.cropImageIfRequested(I, roi, useROI);

% Find the nearest training image size.
[info.PreprocessedImageSize,info.ScaleX,info.ScaleY] = iFindNearestTrainingImageSize(...
    size(Iroi),trainingImageSize);

Ipreprocessed = yolov2ObjectDetector.preprocess(Iroi,info.PreprocessedImageSize);
end

%--------------------------------------------------------------------------
function loader = iCreateDataLoader(ds,miniBatchSize,inputLayerSize)
loader = nnet.internal.cnn.DataLoader(ds,...
    'MiniBatchSize',miniBatchSize,...
    'CollateFcn',@(x)iTryToBatchData(x,inputLayerSize));
end

%--------------------------------------------------------------------------
function data = iTryToBatchData(X, inputLayerSize)
try
    observationDim = numel(inputLayerSize) + 1;
    data{1} = cat(observationDim,X{:,1});
catch e
    if strcmp(e.identifier, 'MATLAB:catenate:dimensionMismatch')
        error(message('vision:ObjectDetector:unableToBatchImagesForDetect'));
    else
        throwAsCaller(e);
    end
end
data{2} = X(:,2:end);
end
