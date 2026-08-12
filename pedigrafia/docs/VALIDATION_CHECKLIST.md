# Checklist de validação física

> Nada neste sistema pode ser chamado de "validado" antes de este roteiro ser
> executado com **hardware real, régua real e impressões reais**. Os testes
> automatizados validam a matemática; este roteiro valida a realidade.

Material: régua/paquímetro metálico (resolução ≥ 0,5 mm, de preferência paquímetro
digital 0,01 mm), o marcador impresso, superfície rígida e plana, impressora A4.

---

## Etapa 0 — Imprimir e conferir o marcador

1. Baixe `GET /api/marker.pdf` (ou pela tela inicial).
2. Imprima em **Tamanho real / 100 %**, com "Ajustar à página" **desligado**.
3. Meça o lado do quadrado preto com paquímetro, nas quatro arestas.

- [ ] Cada aresta mede **50,00 mm ± 0,3 mm**.
- [ ] Se não: a impressora está reescalando. Corrija antes de continuar — **todo o
      resto do sistema herda esse erro proporcionalmente**.
- [ ] Cole o marcador sobre material rígido (PVC, acrílico, papel-cartão grosso).
      Papel ondulado introduz erro de perspectiva local.
- [ ] Mantenha a borda branca (zona de silêncio) ao redor do quadrado.

## Etapa 1 — Posicionar o marcador no plano correto

- [ ] O marcador está **no mesmo plano físico da planta do pé** (sobre o vidro do
      podoscópio, não abaixo dele, não sobre um degrau).
- [ ] O marcador não fica sob o pé nem é coberto por ele.
- [ ] Marcador plano, sem empenamento visível.

> Erro de plano é o erro mais grave possível: 10 mm de diferença de altura entre o
> marcador e a planta, a 700 mm de distância, produz ~1,4 % de erro de escala —
> 3,7 mm em um pé de 265 mm.

## Etapa 2 — Objeto de comprimento conhecido (sem pé)

1. Coloque sobre a plataforma um objeto plano de comprimento **conhecido e medido
   com paquímetro** (ex.: uma régua metálica, uma placa retangular).
2. Fotografe com o marcador visível.
3. Rode a análise e, na tela de **Depuração**, leia o erro de ida-e-volta do marcador.

- [ ] Erro de ida-e-volta < 0,3 mm.
- [ ] Meça o objeto na imagem retificada (ou exporte e meça no PDF).
- [ ] Diferença entre o valor do paquímetro e o do sistema: **registre em mm**.
- [ ] Repita 5 vezes, movendo a câmera entre as capturas.
- [ ] Dispersão entre as 5 medidas: **registre**. Deve ser < 1 mm.

| Repetição | Paquímetro (mm) | Sistema (mm) | Diferença |
|---|---|---|---|
| 1 |  |  |  |
| 2 |  |  |  |
| 3 |  |  |  |
| 4 |  |  |  |
| 5 |  |  |  |

## Etapa 3 — Impressão em escala 1:1

1. Gere o PDF de um pé.
2. Imprima em **Tamanho real / 100 %**.
3. Meça no papel a distância entre os dois pontos extremos do contorno (calcâneo mais
   posterior e pododáctilo mais distal).

- [ ] Valor impresso = valor exibido na tela, **± 1 mm**.
- [ ] Repita em uma segunda impressora, se houver.
- [ ] Meça também a largura do antepé no papel e compare com a tela.

| Medida | Tela (mm) | Papel (mm) | Diferença |
|---|---|---|---|
| Comprimento |  |  |  |
| Largura do antepé |  |  |  |
| Largura do calcâneo |  |  |  |

## Etapa 4 — Pé real contra medição direta

1. Meça o pé do voluntário com um dispositivo de referência (Brannock, régua com
   batente de calcâneo, ou paquímetro sobre papel milimetrado).
2. Fotografe no podoscópio, analise, revise e exporte.

- [ ] Comprimento do sistema vs. medição direta: **registre a diferença**.
- [ ] Repita com pelo menos 5 pessoas, dois pés cada.
- [ ] Calcule média e desvio-padrão das diferenças. **É este número — e não os testes
      sintéticos — que define a exatidão do sistema para uso clínico.**

## Etapa 5 — Reprodutibilidade

- [ ] Mesmo pé, 3 capturas independentes (reposicionando o pé a cada vez).
- [ ] Dispersão entre as 3: registre. Se > 2 mm, revise o protocolo de captura
      (posicionamento do calcâneo, carga sobre o pé, ângulo da câmera).
- [ ] Mesma foto, 2 profissionais revisando de forma independente: dispersão entre as
      geometrias aprovadas. Mede a variabilidade **humana**, que é real e não pode ser
      atribuída ao software.

## Etapa 6 — Lateralidade e orientação do molde

- [ ] O sistema classificou D/E corretamente em todas as capturas.
- [ ] **Recorte o contorno impresso e apoie o pé sobre ele.** O gabarito deve
      encaixar sem espelhamento. Se estiver espelhado, alterne `view` entre
      `below` e `above` e repita.

## Etapa 7 — Quality gate

- [ ] Foto propositalmente desfocada → bloqueada.
- [ ] Foto com o marcador parcialmente coberto → bloqueada.
- [ ] Foto com um pé cortado na borda → bloqueada.
- [ ] Foto boa → aprovada com score ≥ 85.

---

## Registro final

Só após preencher as tabelas acima é legítimo afirmar exatidão. Escreva a conclusão
neste formato, sem arredondar para melhor:

> Em N = ___ medições, contra ___ (método de referência), o sistema apresentou
> diferença média de ___ mm (desvio-padrão ___ mm, máximo ___ mm), nas condições
> ___ (equipamento, iluminação, protocolo).
