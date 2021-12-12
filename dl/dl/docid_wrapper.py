# see https://medium.com/@pennyqxr/how-save-and-load-fasttext-model-in-mlflow-format-37e4d6017bf0

import mlflow
from nnclassifier import NNClassifier
from doc import DocPipeline

class DocWrapper(mlflow.pyfunc.PythonModel):
    """
    Class to train and use doc models
    """

    def load_context(self, context):
        """This method is called when loading an MLflow model with pyfunc.load_model(), as soon as the Python Model is constructed.
        Args:
            context: MLflow context where the model artifact is stored.
        """
        print(context)
        self.model = NNClassifier.load(context.artifacts['classifier_model_path'])
        self.pipeline = DocPipeline.load(context.artifacts['classifier_pipeline_path'])
        self.value_column = 'value'
        print(self.value_column)

    def predict(self, context, model_input):
        """This is an abstract function. We customized it into a method to fetch the FastText model.
        Args:
            context ([type]): MLflow context where the model artifact is stored.
            model_input ([type]): the input data to fit into the model.
        Returns:
            [type]: the loaded model artifact.
        """
        print(self.value_column)
        feature_df = self.pipeline.transform(model_input, self.value_column)
        preds = self.model.predict(feature_df)
        model_input['raw'] = preds
        model_input['prediction'] = (preds > 0).astype(int)
        return model_input