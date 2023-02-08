# Speedy Gonzales: A Collection of Fast Task-Specific Models for Spanish

# Summary

<table>
<thead>
  <tr>
    <th>Model</th>
    <th>Parameters</th>
    <th>Speedup</th>
    <th>Score</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td colspan="4">Fine-tuning</td>
  </tr>
  <tr>
    <td><a href="https://huggingface.co/dccuchile/bert-base-spanish-wwm-uncased" target="_blank" rel="noopener noreferrer">BETO uncased</a></td>
    <td>110M</td>
    <td>1.00x</td>
    <td>81.02</td>
  </tr>
  <tr>
    <td><a href="https://huggingface.co/dccuchile/bert-base-spanish-wwm-cased" target="_blank" rel="noopener noreferrer">BETO cased</a></td>
    <td>110M</td>
    <td>1.00x</td>
    <td>84.82</td>
  </tr>
  <tr>
    <td><a href="https://huggingface.co/dccuchile/distilbert-base-spanish-uncased" target="_blank" rel="noopener noreferrer">DistilBETO</a></td>
    <td>67M</td>
    <td>2.00x</td>
    <td>76.73</td>
  </tr>
  <tr>
    <td><a href="https://huggingface.co/dccuchile/albert-tiny-spanish" target="_blank" rel="noopener noreferrer">ALBETO tiny</a></td>
    <td>5M</td>
    <td>18.05x</td>
    <td>74.97</td>
  </tr>
  <tr>
    <td><a href="https://huggingface.co/dccuchile/albert-base-spanish" target="_blank" rel="noopener noreferrer">ALBETO base</a></td>
    <td>12M</td>
    <td>0.99x</td>
    <td>83.25</td>
  </tr>
  <tr>
    <td><a href="https://huggingface.co/dccuchile/albert-large-spanish" target="_blank" rel="noopener noreferrer">ALBETO large</a></td>
    <td>18M</td>
    <td>0.28x</td>
    <td>82.02</td>
  </tr>
  <tr>
    <td><a href="https://huggingface.co/dccuchile/albert-xlarge-spanish" target="_blank" rel="noopener noreferrer">ALBETO xlarge</a></td>
    <td>59M</td>
    <td>0.07x</td>
    <td>84.13</td>
  </tr>
  <tr>
    <td><a href="https://huggingface.co/dccuchile/albert-xxlarge-spanish" target="_blank" rel="noopener noreferrer">ALBETO xxlarge</a></td>
    <td>223M</td>
    <td>0.03x</td>
    <td>85.17</td>
  </tr>
  <tr>
    <td><a href="https://huggingface.co/bertin-project/bertin-roberta-base-spanish" target="_blank" rel="noopener noreferrer">BERTIN</a></td>
    <td>125M</td>
    <td>1.00x</td>
    <td>83.97</td>
  </tr>
  <tr>
    <td><a href="https://huggingface.co/PlanTL-GOB-ES/roberta-base-bne" target="_blank" rel="noopener noreferrer">RoBERTa BNE base</a></td>
    <td>125M</td>
    <td>1.00x</td>
    <td>84.83</td>
  </tr>
  <tr>
    <td><a href="https://huggingface.co/PlanTL-GOB-ES/roberta-large-bne" target="_blank" rel="noopener noreferrer">RoBERTa BNE large</a></td>
    <td>355M</td>
    <td>0.28x</td>
    <td>68.42</td>
  </tr>
  <tr>
    <td colspan="4">Task-specific Knowledge Distillation</td>
  </tr>
  <tr>
    <td><a href="https://huggingface.co/dccuchile/albert-tiny-spanish" target="_blank" rel="noopener noreferrer">ALBETO tiny</a></td>
    <td>5M</td>
    <td>18.05x</td>
    <td>76.49</td>
  </tr>
  <tr>
    <td><a href="https://huggingface.co/dccuchile/albert-base-2-spanish" target="_blank" rel="noopener noreferrer">ALBETO base-2</a></td>
    <td>12M</td>
    <td>5.96x</td>
    <td>72.98</td>
  </tr>
  <tr>
    <td><a href="https://huggingface.co/dccuchile/albert-base-4-spanish" target="_blank" rel="noopener noreferrer">ALBETO base-4</a></td>
    <td>12M</td>
    <td>2.99x</td>
    <td>80.06</td>
  </tr>
  <tr>
    <td><a href="https://huggingface.co/dccuchile/albert-base-6-spanish" target="_blank" rel="noopener noreferrer">ALBETO base-6</a></td>
    <td>12M</td>
    <td>1.99x</td>
    <td>82.70</td>
  </tr>
  <tr>
    <td><a href="https://huggingface.co/dccuchile/albert-base-8-spanish" target="_blank" rel="noopener noreferrer">ALBETO base-8</a></td>
    <td>12M</td>
    <td>1.49x</td>
    <td>83.78</td>
  </tr>
  <tr>
    <td><a href="https://huggingface.co/dccuchile/albert-base-10-spanish" target="_blank" rel="noopener noreferrer">ALBETO base-10</a></td>
    <td>12M</td>
    <td>1.19x</td>
    <td>84.32</td>
  </tr>
</tbody>
</table>

# All results

<table>
<thead>
  <tr>
    <th rowspan="3">Model</th>
    <th colspan="3" rowspan="2">Text Classification<br>(Accuracy)</th>
    <th colspan="2" rowspan="2">Sequence Tagging<br>(F1 Score)</th>
    <th colspan="3" rowspan="2">Question Answering<br>(F1 Score / Exact Match)</th>
  </tr>
  <tr>
  </tr>
  <tr>
    <th>MLDoc</th>
    <th>PAWS-X</th>
    <th>XNLI</th>
    <th>POS</th>
    <th>NER</th>
    <th>MLQA</th>
    <th>SQAC</th>
    <th>TAR / XQuAD</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td colspan="9">Fine-tuning</td>
  </tr>
  <tr>
    <td>BETO uncased</td>
    <td><a href="https://huggingface.co/dccuchile/bert-base-spanish-wwm-uncased-finetuned-mldoc" target="_blank" rel="noopener noreferrer">96.38</a></td>
    <td><a href="https://huggingface.co/dccuchile/bert-base-spanish-wwm-uncased-finetuned-pawsx" target="_blank" rel="noopener noreferrer">84.25</a></td>
    <td><a href="https://huggingface.co/dccuchile/bert-base-spanish-wwm-uncased-finetuned-xnli" target="_blank" rel="noopener noreferrer">77.76</a></td>
    <td><a href="https://huggingface.co/dccuchile/bert-base-spanish-wwm-uncased-finetuned-pos" target="_blank" rel="noopener noreferrer">97.81</a></td>
    <td><a href="https://huggingface.co/dccuchile/bert-base-spanish-wwm-uncased-finetuned-ner" target="_blank" rel="noopener noreferrer">80.85</a></td>
    <td><a href="https://huggingface.co/dccuchile/bert-base-spanish-wwm-uncased-finetuned-qa-mlqa" target="_blank" rel="noopener noreferrer">64.12 / 40.83</a></td>
    <td><a href="https://huggingface.co/dccuchile/bert-base-spanish-wwm-uncased-finetuned-qa-sqac" target="_blank" rel="noopener noreferrer">72.22 / 53.45</a></td>
    <td><a href="https://huggingface.co/dccuchile/bert-base-spanish-wwm-uncased-finetuned-qa-tar" target="_blank" rel="noopener noreferrer">74.81 / 54.62</a></td>
  </tr>
  <tr>
    <td>BETO cased</td>
    <td><a href="https://huggingface.co/dccuchile/bert-base-spanish-wwm-cased-finetuned-mldoc" target="_blank" rel="noopener noreferrer">96.65</a></td>
    <td><a href="https://huggingface.co/dccuchile/bert-base-spanish-wwm-cased-finetuned-pawsx" target="_blank" rel="noopener noreferrer">89.80</a></td>
    <td><a href="https://huggingface.co/dccuchile/bert-base-spanish-wwm-cased-finetuned-xnli" target="_blank" rel="noopener noreferrer">81.98</a></td>
    <td><a href="https://huggingface.co/dccuchile/bert-base-spanish-wwm-cased-finetuned-pos" target="_blank" rel="noopener noreferrer">98.95</a></td>
    <td><a href="https://huggingface.co/dccuchile/bert-base-spanish-wwm-cased-finetuned-ner" target="_blank" rel="noopener noreferrer">87.14</a></td>
    <td><a href="https://huggingface.co/dccuchile/bert-base-spanish-wwm-cased-finetuned-qa-mlqa" target="_blank" rel="noopener noreferrer">67.65 / 43.38</a></td>
    <td><a href="https://huggingface.co/dccuchile/bert-base-spanish-wwm-cased-finetuned-qa-sqac" target="_blank" rel="noopener noreferrer">78.65 / 60.94</a></td>
    <td><a href="https://huggingface.co/dccuchile/bert-base-spanish-wwm-cased-finetuned-qa-tar" target="_blank" rel="noopener noreferrer">77.81 / 56.97</a></td>
  </tr>
  <tr>
    <td>DistilBETO</td>
    <td><a href="https://huggingface.co/dccuchile/distilbert-base-spanish-uncased-finetuned-mldoc" target="_blank" rel="noopener noreferrer">96.35</a></td>
    <td><a href="https://huggingface.co/dccuchile/distilbert-base-spanish-uncased-finetuned-pawsx" target="_blank" rel="noopener noreferrer">75.80</a></td>
    <td><a href="https://huggingface.co/dccuchile/distilbert-base-spanish-uncased-finetuned-xnli" target="_blank" rel="noopener noreferrer">76.59</a></td>
    <td><a href="https://huggingface.co/dccuchile/distilbert-base-spanish-uncased-finetuned-pos" target="_blank" rel="noopener noreferrer">97.67</a></td>
    <td><a href="https://huggingface.co/dccuchile/distilbert-base-spanish-uncased-finetuned-ner" target="_blank" rel="noopener noreferrer">78.13</a></td>
    <td><a href="https://huggingface.co/dccuchile/distilbert-base-spanish-uncased-finetuned-qa-mlqa" target="_blank" rel="noopener noreferrer">57.97 / 35.50</a></td>
    <td><a href="https://huggingface.co/dccuchile/distilbert-base-spanish-uncased-finetuned-qa-sqac" target="_blank" rel="noopener noreferrer">64.41 / 45.34</a></td>
    <td><a href="https://huggingface.co/dccuchile/distilbert-base-spanish-uncased-finetuned-qa-tar" target="_blank" rel="noopener noreferrer">66.97 / 46.55</a></td>
  </tr>
  <tr>
    <td>ALBETO tiny</td>
    <td><a href="https://huggingface.co/dccuchile/albert-tiny-spanish-finetuned-mldoc" target="_blank" rel="noopener noreferrer">95.82</a></td>
    <td><a href="https://huggingface.co/dccuchile/albert-tiny-spanish-finetuned-pawsx" target="_blank" rel="noopener noreferrer">80.20</a></td>
    <td><a href="https://huggingface.co/dccuchile/albert-tiny-spanish-finetuned-xnli" target="_blank" rel="noopener noreferrer">73.43</a></td>
    <td><a href="https://huggingface.co/dccuchile/albert-tiny-spanish-finetuned-pos" target="_blank" rel="noopener noreferrer">97.34</a></td>
    <td><a href="https://huggingface.co/dccuchile/albert-tiny-spanish-finetuned-ner" target="_blank" rel="noopener noreferrer">75.42</a></td>
    <td><a href="https://huggingface.co/dccuchile/albert-tiny-spanish-finetuned-qa-mlqa" target="_blank" rel="noopener noreferrer">51.84 / 28.28</a></td>
    <td><a href="https://huggingface.co/dccuchile/albert-tiny-spanish-finetuned-qa-sqac" target="_blank" rel="noopener noreferrer">59.28 / 39.16</a></td>
    <td><a href="https://huggingface.co/dccuchile/albert-tiny-spanish-finetuned-qa-tar" target="_blank" rel="noopener noreferrer">66.43 / 45.71</a></td>
  </tr>
  <tr>
    <td>ALBETO base</td>
    <td><a href="https://huggingface.co/dccuchile/albert-base-spanish-finetuned-mldoc" target="_blank" rel="noopener noreferrer">96.07</a></td>
    <td><a href="https://huggingface.co/dccuchile/albert-base-spanish-finetuned-pawsx" target="_blank" rel="noopener noreferrer">87.95</a></td>
    <td><a href="https://huggingface.co/dccuchile/albert-base-spanish-finetuned-xnli" target="_blank" rel="noopener noreferrer">79.88</a></td>
    <td><a href="https://huggingface.co/dccuchile/albert-base-spanish-finetuned-pos" target="_blank" rel="noopener noreferrer">98.21</a></td>
    <td><a href="https://huggingface.co/dccuchile/albert-base-spanish-finetuned-ner" target="_blank" rel="noopener noreferrer">82.89</a></td>
    <td><a href="https://huggingface.co/dccuchile/albert-base-spanish-finetuned-qa-mlqa" target="_blank" rel="noopener noreferrer">66.12 / 41.10</a></td>
    <td><a href="https://huggingface.co/dccuchile/albert-base-spanish-finetuned-qa-sqac" target="_blank" rel="noopener noreferrer">77.71 / 59.84</a></td>
    <td><a href="https://huggingface.co/dccuchile/albert-base-spanish-finetuned-qa-tar" target="_blank" rel="noopener noreferrer">77.18 / 57.05</a></td>
  </tr>
  <tr>
    <td>ALBETO large</td>
    <td><a href="https://huggingface.co/dccuchile/albert-large-spanish-finetuned-mldoc" target="_blank" rel="noopener noreferrer">92.22</a></td>
    <td><a href="https://huggingface.co/dccuchile/albert-large-spanish-finetuned-pawsx" target="_blank" rel="noopener noreferrer">86.05</a></td>
    <td><a href="https://huggingface.co/dccuchile/albert-large-spanish-finetuned-xnli" target="_blank" rel="noopener noreferrer">78.94</a></td>
    <td><a href="https://huggingface.co/dccuchile/albert-large-spanish-finetuned-pos" target="_blank" rel="noopener noreferrer">97.98</a></td>
    <td><a href="https://huggingface.co/dccuchile/albert-large-spanish-finetuned-ner" target="_blank" rel="noopener noreferrer">82.36</a></td>
    <td><a href="https://huggingface.co/dccuchile/albert-large-spanish-finetuned-qa-mlqa" target="_blank" rel="noopener noreferrer">65.56 / 40.98</a></td>
    <td><a href="https://huggingface.co/dccuchile/albert-large-spanish-finetuned-qa-sqac" target="_blank" rel="noopener noreferrer">76.36 / 56.54</a></td>
    <td><a href="https://huggingface.co/dccuchile/albert-large-spanish-finetuned-qa-tar" target="_blank" rel="noopener noreferrer">76.72 / 56.21</a></td>
  </tr>
  <tr>
    <td>ALBETO xlarge</td>
    <td><a href="https://huggingface.co/dccuchile/albert-xlarge-spanish-finetuned-mldoc" target="_blank" rel="noopener noreferrer">95.70</a></td>
    <td><a href="https://huggingface.co/dccuchile/albert-xlarge-spanish-finetuned-pawsx" target="_blank" rel="noopener noreferrer">89.05</a></td>
    <td><a href="https://huggingface.co/dccuchile/albert-xlarge-spanish-finetuned-xnli" target="_blank" rel="noopener noreferrer">81.68</a></td>
    <td><a href="https://huggingface.co/dccuchile/albert-xlarge-spanish-finetuned-pos" target="_blank" rel="noopener noreferrer">98.20</a></td>
    <td><a href="https://huggingface.co/dccuchile/albert-xlarge-spanish-finetuned-ner" target="_blank" rel="noopener noreferrer">81.42</a></td>
    <td><a href="https://huggingface.co/dccuchile/albert-xlarge-spanish-finetuned-qa-mlqa" target="_blank" rel="noopener noreferrer">68.26 / 43.76</a></td>
    <td><a href="https://huggingface.co/dccuchile/albert-xlarge-spanish-finetuned-qa-sqac" target="_blank" rel="noopener noreferrer">78.64 / 59.26</a></td>
    <td><a href="https://huggingface.co/dccuchile/albert-xlarge-spanish-finetuned-qa-tar" target="_blank" rel="noopener noreferrer">80.15 / 59.66</a></td>
  </tr>
  <tr>
    <td>ALBETO xxlarge</td>
    <td><a href="https://huggingface.co/dccuchile/albert-xxlarge-spanish-finetuned-mldoc" target="_blank" rel="noopener noreferrer">96.85</a></td>
    <td><a href="https://huggingface.co/dccuchile/albert-xxlarge-spanish-finetuned-pawsx" target="_blank" rel="noopener noreferrer">89.85</a></td>
    <td><a href="https://huggingface.co/dccuchile/albert-xxlarge-spanish-finetuned-xnli" target="_blank" rel="noopener noreferrer">82.42</a></td>
    <td><a href="https://huggingface.co/dccuchile/albert-xxlarge-spanish-finetuned-pos" target="_blank" rel="noopener noreferrer">98.43</a></td>
    <td><a href="https://huggingface.co/dccuchile/albert-xxlarge-spanish-finetuned-ner" target="_blank" rel="noopener noreferrer">83.06</a></td>
    <td><a href="https://huggingface.co/dccuchile/albert-xxlarge-spanish-finetuned-qa-mlqa" target="_blank" rel="noopener noreferrer">70.17 / 45.99</a></td>
    <td><a href="https://huggingface.co/dccuchile/albert-xxlarge-spanish-finetuned-qa-sqac" target="_blank" rel="noopener noreferrer">81.49 / 62.67</a></td>
    <td><a href="https://huggingface.co/dccuchile/albert-xxlarge-spanish-finetuned-qa-tar" target="_blank" rel="noopener noreferrer">79.13 / 58.40</a></td>
  </tr>
  <tr>
    <td>BERTIN</td>
    <td><a href="https://huggingface.co/dccuchile/bertin-roberta-base-spanish-finetuned-mldoc" target="_blank" rel="noopener noreferrer">96.47</a></td>
    <td><a href="https://huggingface.co/dccuchile/bertin-roberta-base-spanish-finetuned-pawsx" target="_blank" rel="noopener noreferrer">88.65</a></td>
    <td><a href="https://huggingface.co/dccuchile/bertin-roberta-base-spanish-finetuned-xnli" target="_blank" rel="noopener noreferrer">80.50</a></td>
    <td><a href="https://huggingface.co/dccuchile/bertin-roberta-base-spanish-finetuned-pos" target="_blank" rel="noopener noreferrer">99.02</a></td>
    <td><a href="https://huggingface.co/dccuchile/bertin-roberta-base-spanish-finetuned-ner" target="_blank" rel="noopener noreferrer">85.66</a></td>
    <td><a href="https://huggingface.co/dccuchile/bertin-roberta-base-spanish-finetuned-qa-mlqa" target="_blank" rel="noopener noreferrer">66.06 / 42.16</a></td>
    <td><a href="https://huggingface.co/dccuchile/bertin-roberta-base-spanish-finetuned-qa-sqac" target="_blank" rel="noopener noreferrer">78.42 / 60.05</a></td>
    <td><a href="https://huggingface.co/dccuchile/bertin-roberta-base-spanish-finetuned-qa-tar" target="_blank" rel="noopener noreferrer">77.05 / 57.14</a></td>
  </tr>
  <tr>
    <td>RoBERTa BNE base</td>
    <td><a href="https://huggingface.co/dccuchile/roberta-base-bne-finetuned-mldoc" target="_blank" rel="noopener noreferrer">96.82</a></td>
    <td><a href="https://huggingface.co/dccuchile/roberta-base-bne-finetuned-pawsx" target="_blank" rel="noopener noreferrer">89.90</a></td>
    <td><a href="https://huggingface.co/dccuchile/roberta-base-bne-finetuned-xnli" target="_blank" rel="noopener noreferrer">81.12</a></td>
    <td><a href="https://huggingface.co/dccuchile/roberta-base-bne-finetuned-pos" target="_blank" rel="noopener noreferrer">99.00</a></td>
    <td><a href="https://huggingface.co/dccuchile/roberta-base-bne-finetuned-ner" target="_blank" rel="noopener noreferrer">86.80</a></td>
    <td><a href="https://huggingface.co/dccuchile/roberta-base-bne-finetuned-qa-mlqa" target="_blank" rel="noopener noreferrer">67.31 / 44.50</a></td>
    <td><a href="https://huggingface.co/dccuchile/roberta-base-bne-finetuned-qa-sqac" target="_blank" rel="noopener noreferrer">80.53 / 62.72</a></td>
    <td><a href="https://huggingface.co/dccuchile/roberta-base-bne-finetuned-qa-tar" target="_blank" rel="noopener noreferrer">77.16 / 55.46</a></td>
  </tr>
  <tr>
    <td>RoBERTa BNE large</td>
    <td><a href="https://huggingface.co/dccuchile/roberta-large-bne-finetuned-mldoc" target="_blank" rel="noopener noreferrer">97.00</a></td>
    <td><a href="https://huggingface.co/dccuchile/roberta-large-bne-finetuned-pawsx" target="_blank" rel="noopener noreferrer">90.00</a></td>
    <td><a href="https://huggingface.co/dccuchile/roberta-large-bne-finetuned-xnli" target="_blank" rel="noopener noreferrer">51.62</a></td>
    <td><a href="https://huggingface.co/dccuchile/roberta-large-bne-finetuned-pos" target="_blank" rel="noopener noreferrer">61.83</a></td>
    <td><a href="https://huggingface.co/dccuchile/roberta-large-bne-finetuned-ner" target="_blank" rel="noopener noreferrer">21.47</a></td>
    <td><a href="https://huggingface.co/dccuchile/roberta-large-bne-finetuned-qa-mlqa" target="_blank" rel="noopener noreferrer">67.69 / 44.88</a></td>
    <td><a href="https://huggingface.co/dccuchile/roberta-large-bne-finetuned-qa-sqac" target="_blank" rel="noopener noreferrer">80.41 / 62.14</a></td>
    <td><a href="https://huggingface.co/dccuchile/roberta-large-bne-finetuned-qa-tar" target="_blank" rel="noopener noreferrer">77.34 / 56.97</a></td>
  </tr>
  <tr>
    <td colspan="9">Task-specific Knowledge Distillation</td>
  </tr>
  <tr>
    <td>ALBETO tiny</td>
    <td><a href="https://huggingface.co/dccuchile/albert-tiny-spanish-distilled-mldoc" target="_blank" rel="noopener noreferrer">96.40</a></td>
    <td><a href="https://huggingface.co/dccuchile/albert-tiny-spanish-distilled-pawsx" target="_blank" rel="noopener noreferrer">85.05</a></td>
    <td><a href="https://huggingface.co/dccuchile/albert-tiny-spanish-distilled-xnli" target="_blank" rel="noopener noreferrer">75.99</a></td>
    <td><a href="https://huggingface.co/dccuchile/albert-tiny-spanish-distilled-pos" target="_blank" rel="noopener noreferrer">97.36</a></td>
    <td><a href="https://huggingface.co/dccuchile/albert-tiny-spanish-distilled-ner" target="_blank" rel="noopener noreferrer">72.51</a></td>
    <td><a href="https://huggingface.co/dccuchile/albert-tiny-spanish-distilled-qa-mlqa" target="_blank" rel="noopener noreferrer">54.17 / 32.22</a></td>
    <td><a href="https://huggingface.co/dccuchile/albert-tiny-spanish-distilled-qa-sqac" target="_blank" rel="noopener noreferrer">63.03 / 43.35</a></td>
    <td><a href="https://huggingface.co/dccuchile/albert-tiny-spanish-distilled-qa-tar" target="_blank" rel="noopener noreferrer">67.47 / 46.13</a></td>
  </tr>
  <tr>
    <td>ALBETO base-2</td>
    <td><a href="https://huggingface.co/dccuchile/albert-base-2-spanish-distilled-mldoc" target="_blank" rel="noopener noreferrer">96.20</a></td>
    <td><a href="https://huggingface.co/dccuchile/albert-base-2-spanish-distilled-pawsx" target="_blank" rel="noopener noreferrer">76.75</a></td>
    <td><a href="https://huggingface.co/dccuchile/albert-base-2-spanish-distilled-xnli" target="_blank" rel="noopener noreferrer">73.65</a></td>
    <td><a href="https://huggingface.co/dccuchile/albert-base-2-spanish-distilled-pos" target="_blank" rel="noopener noreferrer">97.17</a></td>
    <td><a href="https://huggingface.co/dccuchile/albert-base-2-spanish-distilled-ner" target="_blank" rel="noopener noreferrer">69.69</a></td>
    <td><a href="https://huggingface.co/dccuchile/albert-base-2-spanish-distilled-qa-mlqa" target="_blank" rel="noopener noreferrer">48.62 / 26.17</a></td>
    <td><a href="https://huggingface.co/dccuchile/albert-base-2-spanish-distilled-qa-sqac" target="_blank" rel="noopener noreferrer">58.40 / 39.00</a></td>
    <td><a href="https://huggingface.co/dccuchile/albert-base-2-spanish-distilled-qa-tar" target="_blank" rel="noopener noreferrer">63.41 / 42.35</a></td>
  </tr>
  <tr>
    <td>ALBETO base-4</td>
    <td><a href="https://huggingface.co/dccuchile/albert-base-4-spanish-distilled-mldoc" target="_blank" rel="noopener noreferrer">96.35</a></td>
    <td><a href="https://huggingface.co/dccuchile/albert-base-4-spanish-distilled-pawsx" target="_blank" rel="noopener noreferrer">86.40</a></td>
    <td><a href="https://huggingface.co/dccuchile/albert-base-4-spanish-distilled-xnli" target="_blank" rel="noopener noreferrer">78.68</a></td>
    <td><a href="https://huggingface.co/dccuchile/albert-base-4-spanish-distilled-pos" target="_blank" rel="noopener noreferrer">97.60</a></td>
    <td><a href="https://huggingface.co/dccuchile/albert-base-4-spanish-distilled-ner" target="_blank" rel="noopener noreferrer">74.58</a></td>
    <td><a href="https://huggingface.co/dccuchile/albert-base-4-spanish-distilled-qa-mlqa" target="_blank" rel="noopener noreferrer">62.19 / 38.28</a></td>
    <td><a href="https://huggingface.co/dccuchile/albert-base-4-spanish-distilled-qa-sqac" target="_blank" rel="noopener noreferrer">71.41 / 52.87</a></td>
    <td><a href="https://huggingface.co/dccuchile/albert-base-4-spanish-distilled-qa-tar" target="_blank" rel="noopener noreferrer">73.31 / 52.43</a></td>
  </tr>
  <tr>
    <td>ALBETO base-6</td>
    <td><a href="https://huggingface.co/dccuchile/albert-base-6-spanish-distilled-mldoc" target="_blank" rel="noopener noreferrer">96.40</a></td>
    <td><a href="https://huggingface.co/dccuchile/albert-base-6-spanish-distilled-pawsx" target="_blank" rel="noopener noreferrer">88.45</a></td>
    <td><a href="https://huggingface.co/dccuchile/albert-base-6-spanish-distilled-xnli" target="_blank" rel="noopener noreferrer">81.66</a></td>
    <td><a href="https://huggingface.co/dccuchile/albert-base-6-spanish-distilled-pos" target="_blank" rel="noopener noreferrer">97.82</a></td>
    <td><a href="https://huggingface.co/dccuchile/albert-base-6-spanish-distilled-ner" target="_blank" rel="noopener noreferrer">78.41</a></td>
    <td><a href="https://huggingface.co/dccuchile/albert-base-6-spanish-distilled-qa-mlqa" target="_blank" rel="noopener noreferrer">66.35 / 42.01</a></td>
    <td><a href="https://huggingface.co/dccuchile/albert-base-6-spanish-distilled-qa-sqac" target="_blank" rel="noopener noreferrer">76.99 / 59.00</a></td>
    <td><a href="https://huggingface.co/dccuchile/albert-base-6-spanish-distilled-qa-tar" target="_blank" rel="noopener noreferrer">75.59 / 56.72</a></td>
  </tr>
  <tr>
    <td>ALBETO base-8</td>
    <td><a href="https://huggingface.co/dccuchile/albert-base-8-spanish-distilled-mldoc" target="_blank" rel="noopener noreferrer">96.70</a></td>
    <td><a href="https://huggingface.co/dccuchile/albert-base-8-spanish-distilled-pawsx" target="_blank" rel="noopener noreferrer">89.75</a></td>
    <td><a href="https://huggingface.co/dccuchile/albert-base-8-spanish-distilled-xnli" target="_blank" rel="noopener noreferrer">82.55</a></td>
    <td><a href="https://huggingface.co/dccuchile/albert-base-8-spanish-distilled-pos" target="_blank" rel="noopener noreferrer">97.96</a></td>
    <td><a href="https://huggingface.co/dccuchile/albert-base-8-spanish-distilled-ner" target="_blank" rel="noopener noreferrer">80.23</a></td>
    <td><a href="https://huggingface.co/dccuchile/albert-base-8-spanish-distilled-qa-mlqa" target="_blank" rel="noopener noreferrer">67.39 / 42.94</a></td>
    <td><a href="https://huggingface.co/dccuchile/albert-base-8-spanish-distilled-qa-sqac" target="_blank" rel="noopener noreferrer">77.79 / 59.63</a></td>
    <td><a href="https://huggingface.co/dccuchile/albert-base-8-spanish-distilled-qa-tar" target="_blank" rel="noopener noreferrer">77.89 / 56.72</a></td>
  </tr>
  <tr>
    <td>ALBETO base-10</td>
    <td><a href="https://huggingface.co/dccuchile/albert-base-10-spanish-distilled-mldoc" target="_blank" rel="noopener noreferrer">96.88</a></td>
    <td><a href="https://huggingface.co/dccuchile/albert-base-10-spanish-distilled-pawsx" target="_blank" rel="noopener noreferrer">89.95</a></td>
    <td><a href="https://huggingface.co/dccuchile/albert-base-10-spanish-distilled-xnli" target="_blank" rel="noopener noreferrer">82.26</a></td>
    <td><a href="https://huggingface.co/dccuchile/albert-base-10-spanish-distilled-pos" target="_blank" rel="noopener noreferrer">98.00</a></td>
    <td><a href="https://huggingface.co/dccuchile/albert-base-10-spanish-distilled-ner" target="_blank" rel="noopener noreferrer">81.10</a></td>
    <td><a href="https://huggingface.co/dccuchile/albert-base-10-spanish-distilled-qa-mlqa" target="_blank" rel="noopener noreferrer">68.29 / 44.29</a></td>
    <td><a href="https://huggingface.co/dccuchile/albert-base-10-spanish-distilled-qa-sqac" target="_blank" rel="noopener noreferrer">79.89 / 62.04</a></td>
    <td><a href="https://huggingface.co/dccuchile/albert-base-10-spanish-distilled-qa-tar" target="_blank" rel="noopener noreferrer">78.21 / 56.21</a></td>
  </tr>
</tbody>
</table>
