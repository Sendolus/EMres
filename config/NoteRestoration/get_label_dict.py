def get_dict_of_label_ids():
    
    early_music_folium_dict = get_label_dictionary()
    label_ids = {}
    counter= 0
    for key in early_music_folium_dict.keys():
        for label  in early_music_folium_dict[key]:
            label_ids[label] = counter
            counter +=1
    return label_ids
    

def get_label_dictionary():
    early_music_folium_dict = {
    'non music symbols': ['staff',  'ornate element', 'otherpage','text', 'blank space' ],
        
    'non ligature notes': ['maxima', 'longa', 'colored longa', 'breve', 'colored breve', 'semibreve',
                           'colored semibreve', 'minim', 'semiminim', 'fusa', 'semifusa'],  
        
    'ligature notes': ['l1longa', 'l2longa', 'l1colored longa', 'l2colored longa', 'l1breve', 'l2breve',
                       'l1colored breve', 'l2colored breve', 'l1semibreve', 'l2semibreve', 'l1colored semibreve',\
                       'l2colored semibreve', 'o1longa', 'o1colored longa', 'o2longa', 'o2colored longa', 'o1breve',\
                       'o1colored breve', 'o2breve', 'o2colored breve', 'o1semibreve', 'o1colored semibreve', 'o2semibreve',\
                       'o2colored semibreve'],
    
    'rests' : ['maxima rest', 'longa rest', 'breve rest', 'semibreve rest', 'minim rest',\
             'fusa rest', 'semifusa rest'],
        
    'meters' : ['imaj', 'pmaj', 'imin', 'pmin', 'imincut', 'pmincut', '3', '2', '1' ],
        
    'clefs' : ['c clef', 'f clef', 'g clef'],
        
    'accidentals' : ['flat', 'sharp'],
        
    'other music symbols' : ['dot', 'barline', 'fermata', 'repeat', 'custos', 'congruence'],
        
    'super classes' : ['rest', 'l1', 'l2', 'o1', 'o2', 'colored l1', 'colored o1', 'colored l2', 'colored o2', 'non ligature'] 
    }
    return early_music_folium_dict



    

        