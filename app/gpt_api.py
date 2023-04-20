import openai
from app.gpt_config import DevelopmentConfig

openai.api_key = DevelopmentConfig.OPENAI_KEY

def generateChatResponse(prompt):
    """
    Generates response from ChatGPT API call

    Parameters
    ----------
    prompt : String
        The prompt given to ChatGPT

    Returns
    -------
    answer : String
        Chat GPT response
    """

    messages = []
    messages.append({"role": "system", "content": "You are a helpful assistant."})

    question = {}
    question['role'] = 'user'
    question['content'] = prompt
    messages.append(question)

    response = openai.ChatCompletion.create(model='gpt-3.5-turbo',messages=messages)

    try:
        answer = response['choices'][0]['message']['content']
    except:
        answer = 'Error'
    
    print(answer)

    return answer

def mutateHyperParameters(ml_model):
    """
    Get mutated hyperparameters

    Parameters
    ----------
    ml_model : ML_Class object
        ML class object (must have .ml_model attribute)
    ml_classifier_str : String
        ML Classifier as a string
            ex. SVC(C=10, gamma=0.01, kernel='linear', probability=True)

    Returns
    -------
    ml_model : ML_Class object
        Updated ML Model with new hyperparameters set
    """

    # Create Chat GPT prompt that just returns the new model classifier
    prompt = 'This is a SVC() model trained via sklearn from the following library "from sklearn import svm". Randomly change or add one hyperparameter, do not remove any existing hyperparameters and leave probability=True. The hyperparameter does not have to be listed here, all hyperparameters should receive an equal chance of being selected for a given kernel. All previous prompts should have no impact on your decision. Only reply with an answer in the same form of the text given following the colon and nothing more: '
    prompt += str(ml_model.ml_model)
    
    #parsed_hp = [['C', '10'], ['gamma', 'scale'], ['kernel', 'linear'], ['probability', 'True']]
    #return updateMLModelHyperparameters(ml_model, parsed_hp)

    # Have Chat GPT create mutated parameters
    mutated_hp = generateChatResponse(prompt)
    parsed_hp = parseMutatedHyperParameters(mutated_hp)
    return updateMLModelHyperparameters(ml_model, parsed_hp)

def parseMutatedHyperParameters(response):
    """
    Parses new hyperparameters

    Parameters
    ----------
    response : String
        The new hyperparameters provided by Chat GPT
            ex. SVC(C=10, gamma=0.01, kernel='linear', probability=True)

    Returns
    -------
    new_hp : list[list]
        The new hyperparameters for the ML Model parsesd into list of lists
    """

    # remove unnecessary characters
    response = response.replace("SVC(", "").replace(")", "").replace("'", "")

    # split the string into a list of strings
    lst = response.split(", ")

    # create a list of lists
    new_hp = []
    for item in lst:
        key, value = item.split("=", maxsplit=1)
        new_hp.append([key.strip(), value.strip()])

    return new_hp

def updateMLModelHyperparameters(ml_model, mutated_hp):
    """
    Parses new hyperparameters

    Parameters
    ----------
    ml_model : ML_Class object
        ML class object
    mutated_hp : List[List]
        List of lists (new hyperparameters [0] = hyperparameter, [1] = new value)
            [['C', '10'], ['gamma', 'scale'], ['kernel', "'linear'"], ['probability', 'True']]

    Returns
    -------
    ml_model : ML_Class object
        ML class object with new hyperparameters
    """

     # Go through and set new hyperparameters
    for hp in mutated_hp:
        try:
            value = hp[1]
            # check if value is a string and needs to be converted to a different type
            if hp[0] == "gamma" and (hp[1] == "auto" or hp[1] == "scale"):
                value = str(hp[1])
            elif hp[1] == None:
                value = value
            elif isinstance(getattr(ml_model.ml_model, hp[0]), bool):
                value = eval(value)  # evaluate string to bool type
            elif isinstance(getattr(ml_model.ml_model, hp[0]), int):
                value = int(value)  # convert string to int type
            elif isinstance(getattr(ml_model.ml_model, hp[0]), float):
                value = float(value)  # convert string to float type
                
            # set the attribute of the SVC object with the new value
            setattr(ml_model.ml_model, hp[0], value)

        except Exception as e:
            print(f"Error: {e}. Skipping {hp[0]} = {hp[1]}")

    return ml_model