import { Database } from "../db.js";
import { MLTrainingParams, MLPredictionParams } from "../types.js";
export declare class MLTools {
    private db;
    constructor(db: Database);
    trainLinearRegression(params: MLTrainingParams): Promise<any>;
    trainRidgeRegression(params: MLTrainingParams): Promise<any>;
    trainLassoRegression(params: MLTrainingParams): Promise<any>;
    trainLogisticRegression(params: MLTrainingParams): Promise<any>;
    trainRandomForest(params: MLTrainingParams): Promise<any>;
    trainSVM(params: MLTrainingParams): Promise<any>;
    trainKNN(params: MLTrainingParams): Promise<any>;
    trainDecisionTree(params: MLTrainingParams): Promise<any>;
    trainNaiveBayes(params: MLTrainingParams): Promise<any>;
    predict(params: MLPredictionParams): Promise<any>;
    getModelInfo(model_id?: number): Promise<any[]>;
}
//# sourceMappingURL=ml.d.ts.map