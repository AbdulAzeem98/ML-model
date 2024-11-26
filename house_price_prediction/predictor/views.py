from django.shortcuts import render

# Create your views here.
import pickle
from django.shortcuts import render

# Load the pre-trained model
with open('predictor/house_price_model.pkl', 'rb') as file:
    model = pickle.load(file)

def predict_price(request):
    if request.method == 'POST':
        # Get data from form
        area = float(request.POST['area'])
        bedrooms = int(request.POST['bedrooms'])
        bathrooms = int(request.POST['bathrooms'])
        location_score = float(request.POST['location_score'])
       

        # Make a prediction
        prediction = model.predict([[area, bedrooms, bathrooms, location_score]])
        price = round(prediction[0], 2)

        return render(request, 'predictor/result.html', {'price': price})

    return render(request, 'predictor/form.html')
