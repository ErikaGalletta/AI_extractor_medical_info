# Import the necessary libraries
import pandas as pd
from openai import OpenAI       
import json

# Load the data
df = pd.read_csv("data/transcriptions.csv")
df.head()

## Start coding here, use as many cells as you need
# Initialize the OpenAI client: make sure you have a valid API key named OPENAI_API_KEY in your Environment Variables
client = OpenAI()

function_definition = [{
    
        "name": "extract_medical_info",
        "description": "Extracts age, medical specialty, and recommended treatment from medical text.",
        "parameters": {
            "type": "object", 
            "properties": {
                "age": {
                    "type": "string",
                    "description": "The extracted age from the text."
                },
                "specialty": {
                    "type": "string",
                    "description": "The extracted medical specialty."
                },
                "treatment": {
                    "type": "string",
                    "description": "The recommended treatment extracted from the text."
                }
            }
        },
        "result": {
            "type": "object",
            "properties": {
            "age": {"type": "string", "description": "The extracted age from the text. "},
            "specialty": {"type": "string", "description": "The extracted medical specialty."},
            "treatment": {"type": "string", "description": "The recommended treatment extracted from the text."}
            }
        }
}]


def prompt_creation(df):
    return f""" Please extract the patient's age (just the number value), medical specialty, and recommended treatment from the following text:\n
    Medical Specialty: {df['medical_specialty']},\n Medical Notes: {df['transcription']} \n
    return only the numerical value for age (es. '30-years-old', return '30')."""
age=[]
medical_specialty=[]
treatment=[]
code=[]

for i in range(0, len(df)):
    df_temp=df.iloc[i]
    prompt=prompt_creation(df_temp)
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",# Using GPT-4 for optimal extraction performance
        messages=[{"role": "user", "content": prompt}],
        functions=function_definition
    )
    
    arguments_str = response.choices[0].message.function_call.arguments
    result = json.loads(arguments_str)
    if result.get('age'):
        age.append(result['age'])
    else:
        age.append(None)
        
    if result.get('specialty'):
        
        medical_specialty.append(result['specialty'])  
    else:
        medical_specialty.append(None)
    if result.get('treatment'):
    
        treatment.append(result['treatment'])
        response = client.chat.completions.create(
        model="gpt-3.5-turbo",# Using GPT-4 for optimal extraction performance
        messages=[{"role": "user", "content": f"""Which ICD-10 code does the following recommended treatment correspond to: {result['treatment']}. \n Only return the code itself"""}]
    )
        code.append(response.choices[0].message.content)
    else:
        treatment.append(None)
        code.append(None)
        
    
data = {
'age': age,
'specialty': medical_specialty,
'treatment': treatment,
'ICD_code': code
}

# Convert the dictionary into a DataFrame
df_structured = pd.DataFrame(data)

print(df_structured)
