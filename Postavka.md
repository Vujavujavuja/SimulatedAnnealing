# Zadatak

## Uvod

Simulirano kaljenje (engl. Simulated Annealing) je algoritam za
optimizaciju koji se koristi za pronalaženje približno optimalnih rešenja
za velike i složene probleme. Ovaj algoritam je inspirisan procesom
kaljenja u metalurgiji, gde se metal zagreva i zatim polako hladi kako bi
se poboljšala njegova struktura i smanjila unutrašnja energija.

## Ključne komponente algoritma simuliranog kaljenja

*   Početno rešenje: Algoritam počinje sa nasumičnim rešenjem problema

*   Testiranje susednih rešenja: U svakom koraku algoritm generiše novo
rešenje koje je "blizu" trenutnom. Često je to nasumična modifikacija
trenutnog rešenja.

*   Kriterijum prihvatanja: Na osnovu razlike u "kvalitetu" između
trenutnog i generisanog rešenja, kao i trenutne "temperature", algoritam
odlučuje da li će prihvatiti novo rešenje:
  *    Bolja rešenja se prihvataju
  *    Čak i lošija rešenja mogu biti prihvaćena sa određenom
verovatnoćom, što pomaže u izbegavanju lokalnih minimuma.

*   Temperatura: Predstavlja parametar koji kontroliše verovatnoću
prihvatanja rešenja koja su gora od trenutnog. Visoka temperatura
omogućava algoritmu da istražuje širi spektar rešenja, uključujući i ona
lošija.

*   Hlađenje: Postepeno smanjivanje temperature vodi ka tome da algoritam
postaje sve manje sklon prihvatanju lošijih rešenja. Ovo je analogno
procesu hlađenja u metalurgiji.

*   Ponavljanje: Proces se ponavlja dok temperatura ne dostigne unapred
definisanu minimalnu vrednost ili dok se ne ispune drugi kriterijumi
zaustavljanja (kao što je broj iteracija).

## Primena algoritma
Simulirano kaljenje se koristi u različitim oblastima, uključujući
raspoređivanje zadataka, projektovanje mreža, optimizaciju ruta,
finansijsko modeliranje i mnoge druge složene optimizacione probleme.
Osnovna prednost ovog algoritma je njegova sposobnost da efikasno
pretražuje velike prostore rešenja i pronalazi dobra rešenja za probleme
koji su previše složeni za egzaktna rešenja.


## Simulirano kaljenje i paralelizam

Osnovni algoritam je sekvencijalan, ali postoji nekoliko načina da unesemo
paralelizam.

1.   Paralelizacija procene "kvaliteta" rešenja - obično je u pitanju
računanje neke vrednosti, koje se možda može paralelizovati.

2.   Modifikacija algoritma pri kojoj se u svakoj iteraciji paralelno
testira više susednih rešenja i odabira ono koje je najbolje.

3.   Paralelno pokretanje više paralelnih proces optimizacije (možda sa
različitim početnim stanjima, generatorima nasumičnih brojeva,
podešavanjima vezanim za temperaturu i slično). Opciono, možemo povremeno
proveriti koji proces radi najbolje i odbaciti one lošije.

# Zadatak

Implementirati proces simuliranog kaljenja za problem pronalaženja
"minimalne energije" slike, pri čemu energiju slike računamo kao zbir
apsolutnih razlika između svih horizontalno i vertikalno susednih piksela
u sva 3 kanala boje. Na primer, između dva susedna piksela sa bojama (25,
33, 17) i (77, 47, 0), energija iznosi 204. Slike zapisujemo kao 3 uitn8
matrice (RBG). Testiranja se mogu vršiti polazeći od nasumično generisanih
slika dimanzija 32x32 ili 64x64.

Pri testiranju susednih rešenja, u svakoj iteraciji nasumično biramo jedan
piksel i menjamo mu mesto sa susednim pikselom koji se nalazi desno ili
ispod odabranog piksela.

Ukoliko je novo stanje slike lošije (energija je veća) verovatnoću
prihavtanja promene računamo kao $P = 2^{\frac{-dE}{T_t}}$, pri čemu je
$dE$ promena energije sistema, a $T_t$ trenutna temperatura. Koristiti
linearno hlađenje sistema ($T_t = T_s \cdot (1 - \frac{i}{i_{max}})$, pri
čemu su $T_s$ početna temperatura, $i$ broj trenutne iteracije, a
$i_{max}$ ukupan broj iteracija). Testirati početne temperature u rasponu
10 do 1000. Za sliku dimenzija 32x32 potrebno je oko 30 miliona iteracija
(oko 20 minuta uz efikasnu numpy CPU implementaciju), mada se rad
algoritma primećuje znatno ranije.

Implementacija treba da bude u CUDA / PyCUDA okruženju (čista CUDA sa C
host kodom se takođe priznaje, ali je verovatno teže). Implementriai tri
nivoa paralelzima:

1. Niti duž x dimenzije bloka koristiti za računanje promene energije
slike prilikom zamene. Računati na osnovu najviše 3x4 ili 4x3 podmatrice
oko elementa koji se pomera (u zavisnoti od toga da li se menja sa susedom
desno ili ispod). Nije prihvatljivo ponovno računanje energije cele
matrice u svakoj iteraciji. Energije pojedinačnih piksela (pre izmene) ne
treba čuvati, već se ponovo računaju.
  * Svaka od 12 niti u bloku računa energiju jednog piksela (aposlutnu
razliku od piskela desno i piksela ispod) pre i posle zamene mesta.
(Pomeraj neće uticati na energije svih 12 elemenata, ali je zbog
jednostavnosti dozvoljno računati ih.). Rezultat upisuje u deljenu
memoriju.
  * Ukupnu promenu energije je dozvoljeno računati upotrebom jedne niti
(na primer niti 0)

2. Niti duž y dimenzije bloka koristit za računanje više alternativih
pomeraja, od kojih će najbolji biti izabran. Na primer, ako radimo sa
blokom dimenzija (12, 8, 1), u jednoj iteraciji testiraćemo 8 mogućih
pomeraja (8 piksela, pri čemu svaki pomeramo ili desno ili dole).
Kad izračunamo svih 8 kandidat rešenja, biramo ono koje ima najmanju
energiju. Dozvoljeno je ovo uraditi koristeći samo jednu nit (na primer
nit (0, 0)) u bloku.
    * Matricu sa vrednostima piksela držati u deljenoj memoriji. Voditi
računa o tome da je deljena memorija zajednička na nivou bloka, pa je
fizičke zamene piksela moguće izvršiti samo na kraju dok niti u tački 1
moraju da izvrše izračunavanje bez izmena deljene matrice.
    * Matricu držati kao unit8 (unsigned char), ali voditi paziti na
overflow pri računanju razlika između susednih piksela.

3. Niti različitih blokova izvršavaju paralelene, nezavisne procese
optimizacije. Posle određenog broja iteracija (10 do 100 hiljada) kernel
funkcije se završavaju i ponovo pokreću, pri čemu svi blokovi nastavljaju
od najboljeg međurezultata.

Nizove piksela koji se menjaju, piksela sa kojom se menjaju i nasumičnih
brojeva koji se koriste za odluku o prihvatanju je dozvoljeno unapred
generisati u python kodu i proslediti kroz globalnu memoriju.


## Stavke

1. Kopirati matricu slike iz globalne u deljenu memoriju, upotrebom što je
više moguće niti u bloku. (3 poena)

2. Izračunati energiju početne matrice, upotrebom što je više moguće niti
u bloku. Rezultat smestiti u deljenu memoriju. Koristiti atomicAdd na
nivou bloka ili redukciju (5 poena).

3. Implementirati proces SK koristićei 12 niti duž X dimenzije bloka, pri
čemu je Y dimenzija 1 (testira se samo jedno rešenje u svakoj iteraciji,
kao klasičan SK) (10 poena).

4. Dodati testiranje više rešenja u iteraciji (upotrebom Y dimenzije
bloka) i odabir najboljeg. (4 poena)

5. Dodati više paralenih procesa optimizacije, zaustavljanje kernela i
nastavak od najboljeg razultata (koisti se više blokova). (4 poena)


## Python/Numpy sekvencijalno rešenje
Napomena: CUDA implementacija nije direktan port datog python koda (neke stvari moraju da se drugačije osmisliti). Kod ispod služi kako bi bili sigurni da ste razumeli algoritam, a može poslužiti i za proveru ispravnosti rada CUDA rešenja (očekujemo vizuelno slučne rezultate).

```python
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import numpy as np
import math
import random


def energy(image):
  distances = np.zeros_like(image, dtype=np.float32)
  distances[:, :-1, :] += np.abs(image[:, :-1, :] - image[:, 1:, :])
  distances[:-1, :, :] += np.abs(image[:-1, :, :] - image[1:, :, :])
  return distances.sum()


def energy_delta(image, image2, src, tgt):
  row_low = max(0, src[0] - 1)
  row_high = tgt[0] + 2
  col_low = max(0, src[1] - 1)
  col_high = tgt[1] + 2
  old = energy(image[row_low:row_high, col_low:col_high])
  new = energy(image2[row_low:row_high, col_low:col_high])
  return new - old

moves = np.array([(0, 1), (1, 0)], dtype=np.uint8)


image = np.random.randint(0, 255, (32, 32, 3), dtype=np.int16)
image2 = np.copy(image)
current_energy = energy(image)

plt.imshow(image)
plt.axis('off')
plt.show()

#---------------
starting_temp = 100
total = 30_000_000
swaps = 0

for iteration in tqdm(range(total)):
  t = iteration / total
  temp = (1 - t) * starting_temp
  src = np.random.randint(0, image.shape[0]-1, 2)
  move = moves[np.random.randint(0, 2)]
  tgt = src + move
  image2[src[0], src[1]] = image[tgt[0], tgt[1]]
  image2[tgt[0], tgt[1]] = image[src[0], src[1]]

  dE = energy_delta(image, image2, src, tgt)
  if dE < 0 or random.random() < np.exp2(-dE/temp):
    image[src[0], src[1]] = image2[src[0], src[1]]
    image[tgt[0], tgt[1]] = image2[tgt[0], tgt[1]]
    # current_energy = new_energy
    current_energy += dE
    swaps += 1
  else:
    image2[src[0], src[1]] = image[src[0], src[1]]
    image2[tgt[0], tgt[1]] = image[tgt[0], tgt[1]]

#---------------  

plt.imshow(image)
plt.axis('off')
plt.show()
```
