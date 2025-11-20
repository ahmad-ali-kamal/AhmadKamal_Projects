import matplotlib.pyplot as plt

def explore_data(df):
    print(df.info())
    print(df.describe())
    print(df['label'].value_counts())

    df['label'].value_counts().plot(kind='bar', title='Label Distribution')
    plt.xlabel('Label')
    plt.ylabel('Count')
    plt.show()

    review_lengths = df['review'].apply(len)
    plt.hist(review_lengths, bins=50)
    plt.title('Review Length Distribution')
    plt.xlabel('Length of Review (chars)')
    plt.ylabel('Frequency')
    plt.show()

    print("Sample positive review:\n", df[df['label']==1]['review'].iloc[0])
    print("Sample negative review:\n", df[df['label']==0]['review'].iloc[0])
