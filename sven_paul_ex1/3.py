LIMIT = 2e6

sieve = {n:True for n in range(int(LIMIT))}

x = 2

primes = []

while x**2 < LIMIT:
    #Iter through and cross out multiples
    i = x
    while i <= LIMIT:
        i += x
        sieve[i] = False

    primes.append(x)

    #Find next larger prime
    x += 1
    while not sieve[x]:
        x+=1

print("Sum of primes smaller than {} is {}".format(LIMIT,sum(primes)))
#Sum of primes smaller than 2000000.0 is 141676
