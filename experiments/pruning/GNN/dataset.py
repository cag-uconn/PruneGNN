
from torch_geometric.datasets import Reddit, NELL, Flickr, Planetoid, Yelp, PPI
import torch_geometric.transforms as T

def get_dataset(args, dataset_dir):

    if args.dataset_name == 'Cora' or args.dataset_name == 'Pubmed' or args.dataset_name == 'CiteSeer':
        dataset = Planetoid(dataset_dir, args.dataset_name, transform=T.NormalizeFeatures())
        data = dataset[0]

        num_features = dataset.num_features
        num_classes = dataset.num_classes

    if args.dataset_name == 'Flickr':
        dataset = Flickr(dataset_dir, transform=T.NormalizeFeatures())
        data = dataset[0]

        num_features = dataset.num_features
        num_classes = dataset.num_classes
          

    if args.dataset_name == 'Yelp':
        dataset = Yelp(dataset_dir,  transform=None)
        data = dataset[0]

        num_features = dataset.num_features
        num_classes = dataset.num_classes
            

    if args.dataset_name == 'NELL':
        dataset = NELL(dataset_dir, pre_transform=None)
        data = dataset[0]
        data.x = data.x.to_dense()[:, :5414]
        num_features = 5414
        num_classes = dataset.num_classes

    return dataset, data, num_features, num_classes
