from django.shortcuts import render, redirect
import joblib
import pandas as pd

def predict_result(request):
    if request.method == 'POST':
        # Extract data from the request
        ssc_p = float(request.POST['ssc_p'])
        ssc_b = int(request.POST['ssc_b'])
        hsc_p = float(request.POST['hsc_p'])
        hsc_b = int(request.POST['hsc_b'])
        hsc_s = int(request.POST['hsc_s'])
        degree_p = float(request.POST['degree_p'])
        degree_t = int(request.POST['degree_t'])
        
        # Convert 'workex' to a boolean (0 for 'No', 1 for 'Yes')
        workex = 1 if request.POST['workex'].lower() == 'yes' else 0
        
        etest_p = float(request.POST['etest_p'])

        # Create a dictionary with the input data
        input_data = {
            'ssc_p': ssc_p,
            'ssc_b': ssc_b,
            'hsc_p': hsc_p,
            'hsc_b': hsc_b,
            'hsc_s': hsc_s,
            'degree_p': degree_p,
            'degree_t': degree_t,
            'workex': workex,
            'etest_p': etest_p,
        }

        import os

        # Get the current directory of the script
        current_directory = os.path.dirname(os.path.realpath(__file__))

        # Define the relative path to the model file
        relative_model_path = 'NoteBook/model_campus_placement'

        # Construct the absolute path to the model file
        model_file_path = os.path.join(current_directory, relative_model_path)

        # Replace backslashes with forward slashes in the path
        model_file_path = model_file_path.replace('\\', '/')

        print("Model File Path:", model_file_path)


        # model_file_path = 'E:/downloads/placement_new/placement/my_app/NoteBook/model_campus_placement'



        print("Model file path:", model_file_path)


        model = joblib.load(model_file_path)

        # Make predictions
        input_data_df = pd.DataFrame(input_data, index=[0])
        prediction = model.predict(input_data_df)
        probability = model.predict_proba(input_data_df)

        result = 'Placed' if prediction == 1 else 'Not Placed'
        probability_placed = probability[0][1] * 100

        # Redirect to the result page with the predicted result as parameters
        return render(request, 'result.html', {
            'result': result,
            'probability_placed': probability_placed,
        })

    # Handle the GET request by rendering the initial form
    else:
        return redirect('index')  # Redirect back to the index page if not a POST request

def index(request):
    # Add any necessary context data for the index page here
    return render(request, 'index.html')
