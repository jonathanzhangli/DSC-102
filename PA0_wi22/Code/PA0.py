

def PA0(user_reviews_csv):
    client = Client()
	client = client.restart()
    
    #######################
    # YOUR CODE GOES HERE #
    #######################
    
    # Change <YOUR_USERS_DATAFRAME> to the dataframe variable in which you have the final users dataframe result
    submit = <YOUR_USERS_DATAFRAME>.describe().compute().round(2)    
    with open('results_PA0.json', 'w') as outfile: json.dump(json.loads(submit.to_json()), outfile)
