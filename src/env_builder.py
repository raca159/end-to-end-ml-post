'''
Script dedicado ao processo de criar o ambiente necessário para a execução do treinamento e endpoint para inferencia.
'''
import os
import yaml
import warnings
from typing import Dict, Any, Optional
import requests
from azure.identity import DefaultAzureCredential
from azure.mgmt.resource import ResourceManagementClient
from azure.storage.blob import BlobServiceClient, ContainerClient
from azure.mgmt.resource.resources.models import DeploymentMode
from azure.ai.ml import MLClient, Input, command
from azure.ai.ml.constants import AssetTypes, InputOutputModes
from azure.ai.ml.entities import (
    Workspace,
    Environment,
    AmlCompute,
    Model,
    Job,
    ManagedOnlineEndpoint,
    ManagedOnlineDeployment,
    CodeConfiguration
)

# Função para criar ou atualizar um Resource Group
def get_or_create_rg(resource_group_name: str, location: str, subscription_id: str):
    ''''
    Função projetada para criar ou atualizar um grupo de recursos usando as credenciais da Azure. Neste caso, usaremos o Service Principal via credenciais de Aplicativo, que requerem as seguintes variáveis de ambiente: AZURE_TENANT_ID, AZURE_CLIENT_ID, AZURE_CLIENT_SECRET.
    Recebe como entrada o nome do grupo de recursos a ser criado, a localização onde o recurso deve ser criado e o ID da assinatura da Azure.
    '''
    credential = DefaultAzureCredential()
    # Obtém o objeto de gerenciamento de recursos
    resource_client = ResourceManagementClient(credential, subscription_id)

    # Provisiona o grupo de recursos
    _ = resource_client.resource_groups.create_or_update(
        resource_group_name, {"location": location}
    )

# Função para criar uma conta de armazenamento
def create_storage_account(storage_account_name: str, subscription_id: str, resource_group_name: str, location: str):
    ''''
    Função projetada para criar a conta de armazenamento usando as credenciais da Azure. Esta função usa um modelo ARM padrão como ponto de partida, em seguida, altera os campos relevantes.

    Recebe como entrada o nome da conta de armazenamento a ser criada, o ID da assinatura da Azure, o nome do grupo de recursos onde a conta de armazenamento deve ser criada e a localização onde o recurso deve ser criado.
    '''
    def add_property(template: Dict[str, Any], resource_name: str, property_name: str, property_value: Any) -> None:
        """Adiciona/Atualiza propriedade do ARM Template"""
        if 'resources' in template:
            for resource in template['resources']:
                if 'name' in resource and resource['name'] == resource_name:
                    if 'properties' in resource:
                        resource['properties'][property_name] = property_value
                    else:
                        resource['properties'] = {property_name: property_value}
                    break
    
    credential = DefaultAzureCredential()
    resource_client = ResourceManagementClient(credential, subscription_id)
    template_uri = "https://raw.githubusercontent.com/Azure/azure-quickstart-templates/master/quickstarts/microsoft.storage/storage-account-create/azuredeploy.json"

    # Download template
    template_response = requests.get(template_uri)
    template = template_response.json()
    # Adicionando no output a connection string
    template['outputs']["storageAccountConnectionString"] = {
        "type": "string",
        "value": "[concat('DefaultEndpointsProtocol=https;AccountName=', parameters('storageAccountName'), ';AccountKey=', listKeys(resourceId('Microsoft.Storage/storageAccounts', parameters('storageAccountName')), providers('Microsoft.Storage', 'storageAccounts').apiVersions[0]).keys[0].value)]"
    }

    # Adicionando acesso anônimo ao blob com 'allowBlobPublicAccess'. Isso apenas facilita a leitura no ML Studio, não é necessário.
    # Colocando um aviso aqui para notificar durante execução
    warnings.warn('Adicionando acesso anônimo ao blob.')
    add_property(
        template=template,
        resource_name="[parameters('storageAccountName')]",
        property_name='allowBlobPublicAccess',
        property_value=True
    )

    rg_deployment_result = resource_client.deployments.begin_create_or_update(
        resource_group_name,
        "exampleDeployment",
        {
            "properties": {
                "template": template,
                "parameters": {
                    "location": {
                        "value": location
                    },
                    'storageAccountName': {
                        'value': storage_account_name
                    }
                },
                "mode": DeploymentMode.incremental
            }
        }
    )
    return rg_deployment_result.result()

# função para adquirir cliente para subir dados para o blob
def get_blob_service_client_connection_string(conn_string_storage_account: str) -> BlobServiceClient:
    '''
    Esta função obtém o Cliente Blob a partir da string de conexão para criar e enviar dados para a Azure.
    '''
    # Cria o objeto BlobServiceClient
    blob_service_client = BlobServiceClient.from_connection_string(conn_string_storage_account)
    return blob_service_client

# função para criar ou selecionar o blob container
def create_blob_container(blob_service_client: BlobServiceClient, container_name: str) -> ContainerClient:
    '''
    Esta função usa o Cliente Blob para criar um container com o nome do container passado como parâmetro.
    Pense em um container como uma pasta onde armazenaremos nossos arquivos.

    Esta função retorna um cliente de contêiner usado para manipular o upload de dados para o contêiner.
    Vamos usá-lo para fazer o upload do conjunto de dados que escolhemos.
    '''
    try:
        container_client = blob_service_client.create_container(name=container_name)
    except Exception as e:
        # Assumindo que é devido ao recurso já existir
        print('Erro ao criar o contêiner: {}'.format(e))
        container_client = blob_service_client.get_container_client(container_name)
    return container_client

# função para upload do arquivo para o blob
def upload_blob_file(filepath: str, blobname: str, container_client: ContainerClient):
    '''
    Esta função recebe o caminho do arquivo no computador e o nome de arquivo desejado para ser usado no Azure Blob Container.
    '''
    with open(file=os.path.join(filepath), mode="rb") as data:
        _ = container_client.upload_blob(name=blobname, data=data, overwrite=True)

# função para criar ambiente do Azure Machine Learning
def get_mlclient_and_create_workspace(subscription_id: str, resource_group: str, location: str, workspace: str, create_workspace: bool=True) -> MLClient:
    '''
    Esta função cria um Azure Machine Learning Workspace com o nome fornecido como parâmetro.
    Recebe o ID da subscrição, o nome do grupo de recursos e a localização como parâmetros para determinar onde criar o recurso.
    '''
    # Obtemos um identificador para o workspace
    ml_client = MLClient(DefaultAzureCredential(), subscription_id, resource_group, workspace)

    if create_workspace:
        ws_basic = Workspace(
            name=workspace,
            location=location,
            display_name=workspace
        )
        ws_basic = ml_client.workspaces.begin_create(ws_basic).result()

    return ml_client

# função para criar um ambiente, equivalente a uma imagem docker, para rodar o treinamento e inferencia
def get_or_create_enviroment(ml_client: MLClient, conda_yaml_path: str, base_image: str, description: str ='Custom Enviroment'):
    
    ''''
    Esta função usa o cliente do ML para criar ou atualizar um Environment usando o conda no caminho `conda_yaml_path` e a imagem `base_image`.
    Após a execução, ela iniciará o processo de `build` da imagem personalizada no Azure Machine Learning.
    '''

    # Obtém o nome do ambiente definido no arquivo yaml
    with open(conda_yaml_path, 'r') as f:
        conda_dict = yaml.safe_load(f)
    env_name = conda_dict['name']
    
    # Cria um objeto Environment
    job_env = Environment(
        name=env_name,
        description=description,
        conda_file=conda_yaml_path,
        image=base_image,
    )
    # Cria ou atualiza o Environment
    job_env = ml_client.environments.create_or_update(job_env)
    return job_env

# função para criar o compute cluster utilizando as configurações passadas como argumento
def get_or_create_compute_cluster(ml_client: MLClient, cluster_config: Dict[str, Any]):
    '''
    Esta função recebe o cliente do ML para obter e/ou criar os recursos de computação.
    Também recebe um dicionário contendo informações sobre o cluster a ser construído.

    Inicialmente, a função tenta obter o recurso de computação, assumindo que ele já existe. Se isso não for possível, o cluster é criado.
    '''
    try:
        return ml_client.compute.get(name=cluster_config['name'])
    except Exception as e:
        print('Erro ao obter o cluster de computação: {}. cluster será criado'.format(e))
        pass

    # Configuração do cluster
    cluster_basic = AmlCompute(**cluster_config)
    # Cria ou atualiza o cluster no ML Studio via cliente
    return ml_client.begin_create_or_update(cluster_basic).result()

# função para adquirir o URI dos dados no blob para passar para o Job de treinamento de modelo
def get_datafile_in_blob_for_job_sweep(
        containername: str,
        accountname: str,
        filename: str,
        folder: Optional[str]=None,
    ):
    '''
    Esta função recebe as informações da conta de armazenamento até chegar ao arquivo desejado em um blob.
    Com o URI construído, ela define o AssetType; neste caso, estamos trabalhando com um arquivo.
    Em seguida, ela define o InputOutputMode, já que só estaremos lendo os dados, montaremos os dados como somente leitura (Read-Only).
    '''
    if folder is None:
        path = f'wasbs://{containername}@{accountname}.blob.core.windows.net/{filename}'
    else:
        path = f'wasbs://{containername}@{accountname}.blob.core.windows.net/{folder}/{filename}'
    data_type = AssetTypes.URI_FILE
    mode = InputOutputModes.RO_MOUNT
    return Input(type=data_type, path=path, mode=mode)

# função para lançar a busca de hyperparameteros
def launch_hyperparam_search(
        command_str: str, inputs: Dict[str, Any],
        sweep_inputs: Dict[str, Any], experiment_name: str,
        ml_client: MLClient, compute: str, environment: str,
        sampling_algorithm: str="bayesian", primary_metric: str="Score",
        goal: str="maximize", max_total_trials: int=12, max_concurrent_trials: int=2,
        code: str="./src"
    ):
    '''
    Esta função lança a busca de hiperparâmetros. Observe que ela possui parâmetros adicionais definidos por padrão.
    O `sampling_algorithm` dita como a busca de hiperparâmetros explorará os parâmetros dentro do espaço de busca.
    Os parâmetros primary_metric e goal estão associados à métrica registrada durante o treinamento que deve ser usada como referência para a melhor execução.
    Os parâmetros max_total_trials e max_concurrent_trials lidam com a quantidade de jobs que serão executados durante a busca de hiperparâmetros.
    Os parâmetros experiment_name, ml_client, compute, environment estão relacionados aos parâmetros do ambiente.
    '''

    # Primeiro, criamos um job com o comando bash, as informações de input e as informações do ambiente
    job = command(
        inputs=inputs, compute=compute, environment=environment,
        code=code, command=command_str, experiment_name=experiment_name,
        display_name=experiment_name,
    )
    # Em seguida, aplicamos o espaço de busca de hiperparâmetros ao Job
    job_for_sweep = job(**sweep_inputs)

    # Depois, definimos os parâmetros da busca de hiperparâmetros
    sweep_job = job_for_sweep.sweep(
        sampling_algorithm=sampling_algorithm,
        primary_metric=primary_metric, goal=goal,
        max_total_trials=max_total_trials,
        max_concurrent_trials=max_concurrent_trials
    )
    # E finalmente, criamos a execução e retornamos o job em execução para que possamos monitorá-lo dentro do Python
    returned_sweep_job = ml_client.create_or_update(sweep_job)
    returned_sweep_job = ml_client.jobs.get(name=returned_sweep_job.name)
    return returned_sweep_job

# função para registrar melhor modelo do Experiment
def register_best_model_from_sweep(ml_client: MLClient, returned_sweep_job: Job, model_name: str, register_name="model-test") -> Model:
    '''
    Esta função utiliza o ml client e o Job de busca de hiperparâmetros para selecionar a melhor execução usando o `best_child_run_id` do Job.
    Com a melhor execução, podemos acessar o modelo na pasta `outputs/artifacts` da execução e selecionar a pasta com o `model_name`.
    Observe que o `model_name` precisa corresponder ao nome do modelo usado no script de treinamento.

    Com isso, registramos o modelo sob o `register_name` e agora ele pode ser acessado na guia Model no Azure Machine Learning.
    '''
    # model_name -> passado como comando do job
    if returned_sweep_job.status == "Completed":
        # Primeiro, obtemos a execução que nos deu o melhor resultado
        best_run = returned_sweep_job.properties["best_child_run_id"]
        # Obtemos o modelo desta execução
        model = Model(
            path="azureml://jobs/{}/outputs/artifacts/paths/{}/".format(
                best_run, model_name
            ),
            name=register_name,
            description="Modelo criado a partir da execução.",
            type="custom_model",
        )
        registered_model = ml_client.models.create_or_update(model=model)
    else:
        print('A busca ainda está em andamento.')
        registered_model = None
    return registered_model

# função para criar endpoint
def create_endpoint_specs(ml_client: MLClient, endpoint_name: str, auth_mode: str="key", **kwargs):
    '''
    Esta função utiliza o ml client para criar um endpoint com o `endpoint_name` e o modo de autenticação por chave.
    '''
    # Criar um endpoint online
    endpoint = ManagedOnlineEndpoint(
        name=endpoint_name,
        auth_mode=auth_mode,
        **kwargs
    )
    return ml_client.online_endpoints.begin_create_or_update(endpoint).result()

# função para dar deploy do endpoint usando o melhor modelo e o Enviroment construido.
def get_or_create_endpoint_deployment(
        ml_client: MLClient, deployment_name:str, model: Model,
        env: Environment, endpoint_name: str,
        code: str="./src", filepath: str="score.py",
        instance_type: str="Standard_DS3_v2"
    ):
    '''
    Esta função utiliza o ml client com o endpoint criado para criar um deployment com o nome `deployment_name`.
    Precisamos passar o Enviroment e o Model para que o deployment saiba qual modelo executar e qual ambiente usar para a execução.
    code se relaciona à pasta onde o script está e filepath é o nome do arquivo.
    O instance_type é a instância de computação a ser usada durante a inferência.
    '''
    deployment = ManagedOnlineDeployment(
        name=deployment_name, endpoint_name=endpoint_name,
        model=model,
        environment=env,
        code_configuration=CodeConfiguration(code=code, scoring_script=filepath),
        instance_type=instance_type,
        instance_count=1,
    )
    return ml_client.online_deployments.begin_create_or_update(deployment=deployment).result()

# função para deletar permanentemente os recursos de Machine Learning
def ml_workspace_deletion(ml_client: MLClient, workspace: str):
    # Delete permanentemente o Azure Machine Learining Workspace invés de soft delete
    _ = ml_client.workspaces.begin_delete(
        name=workspace,
        permanently_delete=True,
        delete_dependent_resources=False
    ).result()

# função para deletar os recursos criados
def get_and_delete_rg(resource_group_name: str, subscription_id: str):
    credential = DefaultAzureCredential()
    # Obtenha o objeto de gerenciamento de recursos.
    resource_client = ResourceManagementClient(credential, subscription_id)
    poller = resource_client.resource_groups.begin_delete(resource_group_name)
    result = poller.result()
    return result