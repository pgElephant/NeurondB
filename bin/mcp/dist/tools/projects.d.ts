import { Database } from "../db.js";
export declare class ProjectTools {
    private db;
    constructor(db: Database);
    createProject(params: {
        project_name: string;
        model_type: string;
        description?: string;
    }): Promise<any>;
    listProjects(): Promise<any[]>;
    getProjectInfo(project_name: string): Promise<any>;
    trainKMeansProject(params: {
        project_name: string;
        table_name: string;
        vector_col: string;
        num_clusters: number;
        max_iters?: number;
    }): Promise<any>;
    deployModel(params: {
        project_name: string;
        version?: number;
    }): Promise<any>;
    listProjectModels(project_name: string): Promise<any[]>;
}
//# sourceMappingURL=projects.d.ts.map