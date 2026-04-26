import h5py

filename = "examples/libero/libero_object/pick_up_the_alphabet_soup_and_place_it_in_the_basket_demo.hdf5"
with h5py.File(filename, 'r') as f:
    print("Keys:", list(f.keys()))
    demo_key = list(f.keys())[0]  # Get first demo
    print(f"Structure of demo '{demo_key}':")
    def print_attrs(name, obj):
        print(name)
    f[demo_key].visititems(print_attrs)