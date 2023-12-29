'''
Script dedicado ao processo de criar o ambiente necessário para a execução do treinamento e endpoint para inferencia.
'''
import os
import yaml
import warnings
from typing import Dict, Any
import requests
from azure.identity import DefaultAzureCredential
from azure.mgmt.resource import ResourceManagementClient
from azure.storage.blob import BlobServiceClient, ContainerClient
from azure.mgmt.resource.resources.models import DeploymentMode
from azure.ai.ml import MLClient
from azure.ai.ml.entities import (
    Workspace,
    Environment,
    AmlCompute
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