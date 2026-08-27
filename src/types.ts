export type Framework = 'pytorch' | 'tensorflow' | 'huggingface' | 'sklearn' | 'opencv_webcam';

export type TaskType = 'transfer_learning' | 'custom_cnn' | 'inference_only' | 'dataset_pipeline';

export type ModelBackbone = 
  | 'resnet50'
  | 'vit_b_16'
  | 'efficientnet_b0'
  | 'convnext_tiny'
  | 'mobilenet_v3_large'
  | 'custom_cnn_4layer'
  | 'swin_base';

export type DatasetOption = 
  | 'custom_folder'
  | 'cifar10'
  | 'imagenet'
  | 'oxford_pets'
  | 'fashion_mnist'
  | 'flowers102';

export type OptimizerType = 'AdamW' | 'Adam' | 'SGD' | 'RMSprop';

export type DeviceType = 'cuda' | 'mps' | 'cpu';

export interface CodeGeneratorConfig {
  framework: Framework;
  taskType: TaskType;
  modelBackbone: ModelBackbone;
  dataset: DatasetOption;
  numClasses: number;
  imageSize: number;
  batchSize: number;
  epochs: number;
  learningRate: number;
  optimizer: OptimizerType;
  weightDecay: number;
  useAmp: boolean;
  pretrained: boolean;
  freezeBackbone: boolean;
  device: DeviceType;
  includeAugmentation: boolean;
  savePath: string;
}

export interface SampleImage {
  id: string;
  name: string;
  category: string;
  url: string;
  predictedClass: string;
  confidence: number;
  top5: { label: string; prob: number }[];
  gradCamDescription: string;
  heatmapColor: 'fire' | 'emerald' | 'cyan';
}

export interface ModelInfo {
  id: ModelBackbone;
  name: string;
  framework: string;
  type: 'CNN' | 'Vision Transformer' | 'Hybrid';
  parameters: string;
  gflops: string;
  top1Accuracy: string;
  description: string;
  bestFor: string;
  inputDim: string;
}
