# Our main implementation of the paper here
# this version: I need to do it under encryption. Use paillier and be careful at the scale down step.
# This is working with a small work-around. Coma and fix it later, first deal with Json and FastAPI
# Next: learn and add fastapi in the code
import random
import math
import time
import copy
import numpy as np
from math import gcd
import secrets
import sympy
from sympy.ntheory import jacobi_symbol
from gmpy2 import powmod, invert
import phe
from phe.util import invert
from shamir_secret_sharing import ShamirSecretSharingScheme as Shamir
from shamir_secret_sharing_integers import ShamirSecretSharingIntegers as IntegerShamir
from shamir_secret_sharing_integers import mult_list

from fastapi import FastAPI, HTTPException
app = FastAPI()

################################################################################
#### The crypto part ###########################
DEFAULT_KEYSIZE = 2048 # Bit-length of p and q
NUMBER_PLAYERS = 5
CORRUPTION_THRESHOLD = 2
PRIME_THRESHOLD = 20000 # We will check p and q for prime factors up to THRESHOLD. This value should be chosen to minimize runtime.
MAX_ITERATIONS = 10 # Maximum number of times we will check if rq=0. This value should be chosen to minimize runtime.
CORRECTNESS_PARAMETER_BIPRIMALITY = 100 # Probability that public key is not biprime = 2^-(STATISTICAL_SECURITY)
STATISTICAL_SECURITY_SECRET_SHARING = 40 # Statistical security parameter for Shamir over the integers


def generate_shared_paillier_key( keyLength = DEFAULT_KEYSIZE, n = NUMBER_PLAYERS, t = CORRUPTION_THRESHOLD, threshold = PRIME_THRESHOLD, it = MAX_ITERATIONS, correctParamPrime = CORRECTNESS_PARAMETER_BIPRIMALITY, statSecShamir = STATISTICAL_SECURITY_SECRET_SHARING):
    """
    Main code to obtain a Paillier public key N
    and shares of the private key.
    """
    primeLength = keyLength // 2
    length = 2 * (primeLength + math.ceil((math.log2(n))))
    shamirP = sympy.nextprime(2**length)

    smallPrimeTest = True

    if primeLength < math.log(threshold,2):
        threshold = 1

    print('Bit-length of primes: ', primeLength)

    Key = PaillierSharedKey(primeLength, n, t, threshold, it )

    print('Starting generation of p, q and N...')
    success = 0
    counter_smallPrime = 0
    counter_biprime = 0
    startTime = time.time()

    # Here we define the small primes considered in small prime test.
    # The generated p and q are both 3 mod 4, hence we do not have to check for divisibility by 2.
    primeList = [p for p in sympy.primerange(3, threshold + 1)]

    while success == 0:
        # Generate the factors p and q
        pShares = Key.generate_prime_vector(n, primeLength)
        qShares = Key.generate_prime_vector(n, primeLength)

        # Then obtain N
        N = compute_product(pShares, qShares, n, t, shamirP)

        # Test bi-primality of N
        [success, smallPrimeTestFail] = is_biprime(n, N, pShares, qShares, correctParamPrime, smallPrimeTest, primeList)

        if smallPrimeTestFail:
            counter_smallPrime += 1
        elif not(success):
            counter_biprime += 1

    print('Generation successful.')
    print('Number of insuccesfull trials small prime test: ', counter_smallPrime)
    print('Number of insuccesfull trials biprime test: ', counter_biprime)
    print('Elapsed time: ', math.floor(time.time()-startTime), 'seconds.')

    PublicKey = phe.PaillierPublicKey(N)

    # Instatiate secret-sharing scheme for lambda and beta.
    Int_SS_Scheme = IntegerShamir(statSecShamir, N, n, t)

    LambdaShares = obtain_lambda_shares(Int_SS_Scheme, pShares, qShares, N)

    success = 0
    while success == 0:
        # Generate shares of random mask
        BetaShares = share_random_element(N, Int_SS_Scheme)

        # Obtain shares of secret key.
        SecretKeyShares = LambdaShares * BetaShares

        success, theta = compute_theta(SecretKeyShares, N)

    return Key, pShares, qShares, N, PublicKey, LambdaShares, BetaShares, SecretKeyShares, theta

class PaillierSharedKey():
    """
    Class containing relevant attributes and methods of a shared paillier key.
    """
    
    def __init__(self, keyLength, n, t, threshold, iterations):
        # The keyLength is the size (in bits) of the RSA-modulus N. 
        # Note that N is the product of two primes. 
        # Hence, the size of the primes is half the keyLength.
        self.keyLength = keyLength
        self.n = n
        self.t = t
        self.threshold = threshold
        self.iterations = iterations

    def generate_prime_vector(self, n, length):
        """
         Everty party picks a random share in [2^(length-1),2^length -1].
         Party 1 randomly samples an integer equal to 3 modulo 4.
         The other parties sample integers equal to 0 modulo 4.
         The resulting integers are additive shares of an integer equal to 
         3 modulo 4.
         """
        primeArray = {i:1 for i in range(1,n+1)}

        while primeArray[1] % 4 != 3:
            primeArray[1] = 2**(length-1)+secrets.randbits(length-1)
        for i in range(2,n+1):
            while primeArray[i] % 4 != 0:
                primeArray[i] = 2**(length-1)+secrets.randbits(length-1)
        return primeArray


    def decrypt(self, Ciphertext, n, t, PublicKey, SecretKeyShares, theta):
        """
        Decryption functionality (no decoding).
        It is assumed that Ciphertext and PublicKey are class instances from
        the phe library, and that SecretKeyShares is a Shares class instance
        from Shamir secret sharing over the integers.
        NB: Now assuming reconstruction set = {1, ..., t+1}
        NB: this code perfoms both local partial decryption
        AND combination of them.
        The two will need to be separated to securely run the code on several
        machines.
        """
        nFac = math.factorial(n)

        c = Ciphertext.ciphertext()
        N = PublicKey.n
        NSquare = N**2
        shares = SecretKeyShares.shares
        degree = SecretKeyShares.degree

        # NB: Here the reconstruction set is implicity defined, but any 
        # large enough subset of shares will do.
        reconstruction_shares = {key: shares[key] for key in list(shares.keys())[:degree+1]}

        lagrange_interpol_enum = {i: mult_list([j for j in reconstruction_shares.keys() if i != j]) for i in reconstruction_shares.keys()}
        lagrange_interpol_denom = {i: mult_list([(j - i) for j in reconstruction_shares.keys() if i != j]) for i in reconstruction_shares.keys()}
        exponents = [ (nFac * lagrange_interpol_enum[i] * reconstruction_shares[i]) // lagrange_interpol_denom[i] for i in reconstruction_shares.keys() ]

        # Notice that the partial decryption is already raised to the power given
        # by the Lagrange interpolation coefficient
        partialDecryptions = [ powmod(c, exp, NSquare) for exp in exponents ]
        combinedDecryption = mult_list(partialDecryptions) % NSquare

        if (combinedDecryption-1) % N != 0:
            print("Error, combined decryption minus one not divisible by N")

        message = ( (combinedDecryption-1)//N * invert(theta, N) ) %N

        return message

################################################################################
"""
    High-level functions
"""
################################################################################

def compute_product(dict1, dict2, n, t, shamirP):
    """
    dict1 and dict2 are interpreted as additive share vectors.
    Code will convert them to Shamir,
    then compute and reveal their product.
    """
    Scheme = Shamir(shamirP, n, t)
    # Shamir-share each share and add them (addition done in reshare)
    shares1 = reshare(dict1, Scheme)
    shares2 = reshare(dict2, Scheme)
    # Then multiply and open
    shares = shares1 * shares2
    value = shares.reconstruct_secret()
    return value


def is_biprime(n, N, pShares, qShares, statSec, smallPrimeTest, primeList):
    """
    Distributed bi-primality test.
    """

    if smallPrimeTest == True:
        for P in primeList:
            if N % P == 0:
                smallPrimeTestFail = True
                return [False, smallPrimeTestFail]

    smallPrimeTestFail = False

    is_biprime_out = True
    counter = 0
    startTime = time.time()
    while is_biprime_out and counter < statSec:
        testValue = sum([secrets.randbelow(N) for _ in range(n)]) % N
        if jacobi_symbol(testValue, N) == 1:
            is_biprime_out = is_biprime_parametrized(N, pShares, qShares, testValue)
            if is_biprime_out:
                counter += 1
    return [is_biprime_out, smallPrimeTestFail]


def obtain_lambda_shares(SSScheme, pShares, qShares, N):
    """
    Obtain Shamir shares over the integers of lambda = (p-1)(q-1),
    where N=pq and p=sum(pShares), q=sum(qShares)
    """

    lambdaAdditiveShares = {1:N - pShares[1] - qShares[1] +1}
    for i in range(1, SSScheme.n):
        lambdaAdditiveShares[i+1]=-pShares[i+1] - qShares[i+1]

    lambdaShares = reshare(lambdaAdditiveShares, SSScheme)
    return lambdaShares

def compute_theta(SecretKeyShares, N):
    n = len(SecretKeyShares.shares)

    SharesModN = copy.deepcopy(SecretKeyShares)
    SharesModN.shares = { key: (value % N) for key,value in SharesModN.shares.items() }

    y = SharesModN.reconstruct_secret(modulus=N)

    theta = (y * math.factorial(n)**3) % N

    if gcd(theta, N) == 0:
        return 0, None, None

    return 1, theta


################################################################################
"""
    Auxiliary functions
"""
################################################################################

def reshare(inputDict, Scheme):
    """
    Convert additive (n-out-of-n) sharing to secret sharing given by Scheme.
    """
    shares = Scheme.share_secret(0)
    for i in inputDict.keys():
        shares += Scheme.share_secret(inputDict[i])
    return shares


def share_random_element(modulus, Scheme):
    """
    Obtain Shamir secret-sharing of random element mod modulus.
    """
    randomList = [secrets.randbelow(modulus) for _ in range(Scheme.n)]

    randomShares = Scheme.share_secret(randomList[0])
    for i in range(1,Scheme.n):
        randomShares += Scheme.share_secret(randomList[i])
    return randomShares


def is_biprime_parametrized(N, pShares, qShares, testValue):
    """
    Test whether N is the product of two primes.
    If so, this test will succeed with probability 1.
    If N is not the product of two primes, the test will fail with at least 
    probability 1/2.
    """

    values = [ int(powmod(testValue, x//4, N)) for x in [sum(y) for y in zip(list(pShares.values())[1:], list(qShares.values())[1:])] ]
    values = [ int(powmod(testValue, (N - pShares[1] - qShares[1] +1)//4, N)) ] + values
    product = mult_list(values[1:])
    if values[0] % N == product % N or values[0] % N == -product % N:
        # Test succeeds, so N is "probably" the product of two primes.
        return True

    # Test fails, so N is definitely not the product of two primes.
    return False


def mult_list(L):
    out=1
    for l in L:
        out=out*l
    return out
#####################################################################################
#####################################################################################
#####################################################################################

class Client:
    def __init__(self, client_id, Key, pShares, qShares, N, PublicKey, LambdaShares, BetaShares, SecretKeyShares, theta):
        self.client_id = client_id
        self.risk_score = random.randint(100,999) 
        self.transactions = {}  # Transactions with other clients (key=client_id, value=amount)        
        self.enc_score =  PublicKey.encrypt(self.risk_score)
        self.dec_score = 0

    def record_transaction(self, other_client_id, amount):
        """Record transaction with another client (positive for incoming, negative for outgoing)."""
        if other_client_id not in self.transactions:
            self.transactions[other_client_id] = 0
        self.transactions[other_client_id] += amount  # Record the transaction (positive/negative based on amount)

    def get_total_positive_transactions(self):
        """Calculate the sum of positive (incoming) transactions."""
        return sum(amount for amount in self.transactions.values() if amount > 0)

    def get_total_negative_transactions(self):
        """Calculate the sum of negative (outgoing) transactions."""
        return sum(amount for amount in self.transactions.values() if amount < 0)

    def compute_new_score(self, delta, constant, received_amounts, sender_risk_scores, PublicKey):
            """Update the risk score based on received transactions and their senders' scores."""
            total_received_score_contribution = 0
            sumOfInput = 0
            for sender_id, amount in received_amounts.items():
                sumOfInput += amount
                total_received_score_contribution += (sender_risk_scores[sender_id] * amount)

            # New score calculation
            if sumOfInput > 0:
                total_received_score_contributionUpdate = total_received_score_contribution * int((delta/sumOfInput))
            #else:
                #total_received_score_contributionUpdate = total_received_score_contribution * (delta)
                #print("new_score", (self.risk_score * (1000 - delta)) + (total_received_score_contributionUpdate))
                new_score = int(powmod((self.risk_score * (1000 - delta)) + (total_received_score_contributionUpdate), 1, 999))
                self.enc_score = PublicKey.encrypt(new_score)
            return self.enc_score

class Bank:
    def __init__(self, bank_id, num_clients, Key, pShares, qShares, N, PublicKey, LambdaShares, BetaShares, SecretKeyShares, theta):
        self.bank_id = bank_id
        self.clients = [Client(f"C{i}B{bank_id}", Key, pShares, qShares, N, PublicKey, LambdaShares, BetaShares, SecretKeyShares, theta) for i in range(1, num_clients + 1)]

    def get_client(self, client_id):
        """Return a client object by its id."""
        for client in self.clients:
            if client.client_id == client_id:
                return client
        return None

    def simulate_transactions(self, other_banks):
        """Simulate random transactions between clients from different banks."""
        for client in self.clients:
            # Choose a random other bank and a client to interact with
            other_bank = random.choice(other_banks)
            other_client = random.choice(other_bank.clients)

            # Random transaction amount
            amount = random.randint(50, 500)
            
            # Make the transaction: money is leaving the sender (negative) and entering the receiver (positive)
            self.transfer(client.client_id, other_client.client_id, amount, other_bank)

    def transfer(self, sender_id, receiver_id, amount, receiving_bank):
        """Transfer money from one client to another (negative on sender, positive on receiver)."""
        sender = self.get_client(sender_id)
        receiver = receiving_bank.get_client(receiver_id)

        if sender and receiver:
            # Record the transaction in both sender and receiver
            sender.record_transaction(receiver_id, -amount)  # Sender sends money (negative)
            receiver.record_transaction(sender_id, amount)   # Receiver receives money (positive)

    def update_client_scores(self, delta, constant, all_banks, PublicKey):
        """Update the scores for all clients in the bank based on received transactions."""
        sender_risk_scores = {client.client_id: client.risk_score for bank in all_banks for client in bank.clients}
        for client in self.clients:
            received_amounts = {k: v for k, v in client.transactions.items() if v > 0}  # Filter only received (positive) amounts
            client.compute_new_score(delta, constant, received_amounts, sender_risk_scores, PublicKey)
            client.dec_score = int(Key.decrypt(client.enc_score, NUMBER_PLAYERS, CORRUPTION_THRESHOLD, PublicKey, SecretKeyShares, theta))
            

    def get_bank_status(self, Key, NUMBER_PLAYERS, CORRUPTION_THRESHOLD, PublicKey, SecretKeyShares, theta):
        """Get the status of all clients in the bank."""
        status = []
        for client in self.clients:
            total_positive = client.get_total_positive_transactions()
            total_negative = client.get_total_negative_transactions()
            #print("THE RISK SCORE:", client.risk_score)
            client_info = {
                "client_id": client.client_id,
                "risk_score(dec)": client.dec_score,
                #"total_positive_transactions": total_positive,
                #"total_negative_transactions": total_negative,
                #"transactions": client.transactions
            }
            status.append(client_info)
        return status

# Simulate a few iterations before exposing the API (this can be adjusted as needed)
def run_simulation(iterations, Key, pShares, qShares, N, PublicKey, LambdaShares, BetaShares, SecretKeyShares, theta, banks, NUMBER_PLAYERS, CORRUPTION_THRESHOLD, update):
    num_banks = 5
    #banks = []

    # Initialize 5 banks with a random number of clients (between 5 and 10)
    for bank_id in range(1, num_banks + 1):
        num_clients = random.randint(3, 8)
        banks.append(Bank(bank_id, num_clients, Key, pShares, qShares, N, PublicKey, LambdaShares, BetaShares, SecretKeyShares, theta))

    # Simulate the iterations
    for iteration in range(1, iterations + 1):
        print(f"\n--- Iteration {iteration} ---")

        # Random delta for this iteration
        #delta = int(round(random.uniform(0.01, 0.1), 3))*10000
        delta = random.randint(100,999) 
        constant = 0.3

        # Simulate transactions
        for bank in banks:
            other_banks = [b for b in banks if b != bank]
            bank.simulate_transactions(other_banks)

        # Update client scores
        for bank in banks:
          bank.update_client_scores(delta, constant, banks, PublicKey)
          #bank.update_client_scores(delta, constant, banks)
          #print("bank_id:", bank_id, "clients:", bank.get_bank_status(Key, NUMBER_PLAYERS, CORRUPTION_THRESHOLD, PublicKey, SecretKeyShares, theta))
          #update.append(bank.get_bank_status(Key, NUMBER_PLAYERS, CORRUPTION_THRESHOLD, PublicKey, SecretKeyShares, theta))

        for bank in banks:
            bank_snapshots[iteration] = [bank.get_bank_status(Key, NUMBER_PLAYERS, CORRUPTION_THRESHOLD, PublicKey, SecretKeyShares, theta) for bank in banks]
                
# Run initial simulation for 1 iteration before exposing the API
lengths = [1024] # [50, 100, 256, 512, 1024, 2048]
banks = []
num_banks = 5
bank_snapshots = {i: [] for i in range(num_banks)}  # Dictionary with iteration as key

update = []
finalMatrix = {}
for length in lengths:
    Key, pShares, qShares, N, PublicKey, LambdaShares, BetaShares, SecretKeyShares, theta = generate_shared_paillier_key(keyLength = length)
    print(' Key generation complete')

iterations = int(input("Enter the number of iterations to run: "))
run_simulation(iterations, Key, pShares, qShares, N, PublicKey, LambdaShares, BetaShares, SecretKeyShares, theta, banks, NUMBER_PLAYERS, CORRUPTION_THRESHOLD, update)

@app.get("/banks/{bank_id}/iteration/{iteration}")
async def get_bank_status(bank_id: int, iteration: int):
    if iteration not in bank_snapshots:
        raise HTTPException(status_code=404, detail="Iteration not found")
    
    # Find the bank by ID within the snapshot of the specified iteration
    for bank_snapshot in bank_snapshots[iteration]:
        if bank_snapshot[0]['client_id'].endswith(f"B{bank_id}"):  # Check bank ID by client ID
            return {"bank_id": bank_id, "iteration": iteration, "clients": bank_snapshot}
    
    # If bank is not found, return an error
    raise HTTPException(status_code=404, detail="Bank not found in this iteration")
'''
# API to get the status of a specific bank
@app.get("/banks/{bank_id}/iterations/{AskIteration}")
async def get_bank_status(bank_id: int, AskIteration: int):
    # Find the bank by ID
    AskIteration = int(input("After how many iterations you want to get the risk scores? (must be less than total number of iteration) "))
    for bank in banks:
        if bank.bank_id == bank_id:
            if AskIteration <= iterations:
                        bank.get_bank_status(Key, NUMBER_PLAYERS, CORRUPTION_THRESHOLD, PublicKey, SecretKeyShares, theta)
                        print("========================================================")
                        print("update")
                        print(update)
                        return {"bank_id": bank_id, "clients": update[AskIteration]}
            else:
                print("Invalid input, exiting.!")
                return

            #return {"bank_id": bank_id, "clients": bank.get_bank_status(Key, NUMBER_PLAYERS, CORRUPTION_THRESHOLD, PublicKey, SecretKeyShares, theta)}
    
    # If bank is not found, return an error
    raise HTTPException(status_code=404, detail="Bank not found")
'''

