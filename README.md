# AI-Financial-Assistant
Creation of an AI chatbot prototype assisting you with your portfolio management tasks
##In order to make the code work properly, there are some remarks:

* ###The hotness score is an expermental part of the code
  We worked to implement as a function of the AI assistant but
  it required ressource that we did not possess, nevertheless, the top 5 firms are compared 
  sorted by their "hotness score". This score is an weighted average of different factors that were aiming to capture a
  "trend" effect on a certain firm. The factors are listed in the code, the sentiment analysis of the tweets, news feed 
  and reddit were obtained via an API provider for this custom database.
* ###The database consists of hourly data on price, twitter, reddit, news and google trend volumes and a sentiment score analysis for a selection of 21 firms
  The aim was to construct a hotness ranking among those firm to identify the ones scoring the highest, implying that
  they are trending (since people talk about them) but also that there is a positive outcome (because the sentiment 
  analysis has a high score as well). This would be to try and showcase a relationship with high hotness score and 
  higher returns. The available data are limited (not enough resources to do it on our own, we had to go through an API
  provider), thus it limited the scope of our experiment.

* ###The AI Chatbot needs to be initialized and trained the 1st time, the lines required to do so are the following:
  This chat bot was an idea to implement a friendly user interface to our financial assistant. It would take user 
  inputted sentences and figure out what the user wanted him to do. The implementation of the neural network required to
  do the training was relied on a library ("NEURAL INTENTS") which would take care of the set up of NN. The .json file
  is the input that was created in order to train the model to react to user inputted sentences. Then the intent mappings
  would serve as links for the assistant to know what functions to implement when prompted by the user. The following code
  is the way to initialize and train the model:
    

    assistant_AI = GenericAssistant('intents.json', intent_methods=intents_mapping,
                model_name="AI_Financial_Assistant_model")

    assistant_AI.train_model()
    assistant_AI.save_model()
* The following line are commented on the final code since we saved the model into a separate file, thus the Assistant can
  be loaded via this line instead:

      assistant_AI.load_model(model_name="AI_Financial_Assistant_model")
  Feel free to retrain the model using the "intents.json" file linked as well
  
  *There are comments that explains how each of the function work.
  *####Finally, there are different files in the folder of the project
  The AI_Financial_Assistant_model files are the saved trained AI assistant obtained from the 1st training and 
  initialization of the program. Once trained it can be access faster by the loading command. The portfolio file stores 
  the user's portfolio from one connection to the other so when the chatbot is exitted the portfolio stays saved and
  ready for loading. The initial portfolio file is created via these lines of code: 
  

        portfolio = {} 
        with open('portfolio.pkl', 'wb') as f:
          pickle.dump(portfolio, f)
  Then for the loading part, we use these lines of code:
  
    with open('portfolio.pkl', 'rb') as f:
      portfolio = pickle.load(f)
