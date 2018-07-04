# Kor-Sentence-Similarity

## Sentence/Text Similarity for Korean (In Simple Way)

### Models
- Modification of [Siamese recurrent architectures for learning sentence similarity](https://dl.acm.org/citation.cfm?id=3016291)
- Simple ver.
- Char-CNN & MLP for Siamese Networks

### Details
- Data
    * In data, two questions are seperated by '\t'
- Preprocessing
    * Character Level (음소 or 음절)
    * Digits and Specials
    * For eumjeol(Syllable), use frequent 2350

- Configuration
    - `main.py`     : main run file
    - `--epochs`    : # of training epochs
    - `--batch`     : Batch Size
    * `--lr`        : Learning rate
    * `--strmaxlen` : Maximum Limit of String Length
    * `--charsize`  : Vocab Size
    * `filter_num`  : # of Filter of one CNN Filter
    * `--emb`       : Embedding Dimension
    * `--eumjeol`   : Use Eumjeol(Syllable-level) if specified
    * `threshold`   : Threshold to determine Similar or not 
    * `--model`     : Model Selection (CNN, MLP)

### To Run
- Set FC,layer and CNN layers in 'main.py'
- run 'main.py' with arguments as you wish 
