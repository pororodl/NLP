import utils
import matplotlib.pyplot as plt

def show_w2v_word_embedding(model,data:utils.Dataset,path):
    word_emb = model.embeddings.get_weights()[0]
    for i in range(data.num_word):
        c = 'blue'
        try:
            int(data.i2v[i])
        except ValueError:
            c = 'red'
        plt.text(word_emb[i,0],word_emb[i,1],s=data.i2v[i],color=c,weight='bold')
    plt.xlim(word_emb[:,0].min()-.5,word_emb[:,0].max()+.5)
    plt.ylim(word_emb[:,1].min()-.5,word_emb[:,1].max()+.5)
    plt.xticks(())
    plt.yticks(())
    plt.xlabel('embedding dim1')
    plt.ylabel('embedding dim2')
    plt.show()
