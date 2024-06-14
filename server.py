from flask import Flask, request
import pickle
import tenseal as ts

app = Flask(__name__)

context = None
encrypted_params = []

@app.route('/context', methods=['POST'])
def receive_context():
    global context
    context = ts.context_from(request.data)
    print("Received context")
    return "Context received", 200

@app.route('/update', methods=['POST'])
def update_params():
    global encrypted_params

    # Receive encrypted parameters from client
    enc_params = pickle.loads(request.data)
    encrypted_params.append(enc_params)
    print(f"Received parameters: {len(encrypted_params)}")

    # Check if we have received parameters from both clients
    if len(encrypted_params) == 2:
        # Compute average of encrypted parameters
        enc_avg = encrypted_params[0]
        enc_avg += encrypted_params[1]
        enc_avg /= 2
        print("Computed average of parameters")

        # Serialize the averaged encrypted parameters
        response = pickle.dumps(enc_avg.serialize())
        encrypted_params = []  # Reset for the next round
        return response, 200
    
    return "Waiting for more parameters", 200

if __name__ == '__main__':
    app.run()
