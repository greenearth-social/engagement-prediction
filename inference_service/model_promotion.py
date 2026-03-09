from clearml import Model
from clearml.backend_api.session.client import APIClient


def promote_model(model_id: str, new_tag: str = "prod", old_tag: str = "candidate") -> None:
    
    model = Model(model_id=model_id)

    # remove candidate and add prod
    new_model_tags = [tag for tag in model.tags if tag != old_tag]
    new_model_tags.append(new_tag)

    client = APIClient()
    client.models.edit(model=model_id, tags=new_model_tags.append(new_tag))

    # publish candidate:
    model.publish()


def get_prod_model(model_name: str, project_name: str = "Engagement Prediction") -> Model:
    
    published_prod_models = Model.query_models(project_name=project_name, model_name=model_name, only_published=True, tags=["prod"])
    
    if len(published_prod_models) > 1:
        raise ValueError(f"Multiple published prod models found for model name '{model_name}'!")
    if len(published_prod_models) == 0:
        raise ValueError(f"No published prod models found for model name '{model_name}'!")
    
    return published_prod_models[0]