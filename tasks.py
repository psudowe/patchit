from __future__ import print_function

import h5py
import numpy as np

class TaskPatchesCached(object):
    ''' cached version of the TaskPatches 
        -- here a file with cropped detection bounding boxes is read once
            (created with extract_crops_from_detections.py)

        -- drop_colorchannel : if True, one channel is dropped at random (set to 0)
        -- jitter: None or int > 0, the patch position is jittered by a random amount
    '''
    def __init__(self, filename_crops, patch_size, layout=None, jitter=None, drop_colorchannel=False,
                 patch_mean=None, patch_std=None, range_to_0_255=False, patch_margin=0):
        self.patches_per_call = 1
        self.filename_crops = filename_crops
        self.patch_size = patch_size
        self.jitter = jitter
        self.drop_colorchannel = drop_colorchannel

        self.patch_overlap_x = 0
        self.patch_margin = patch_margin

        self.patch_mean = patch_mean
        self.patch_std = patch_std
        self.range_to_0_255 = range_to_0_255

        if layout is None:
            self.layout = np.array([2, 4])
        else:
            self.layout = layout
        print('TaskPatchesCached: patch layout: ', self.layout)

        self.n_patches = np.prod(self.layout)
        self.label_max = self.n_patches

        # read the detections from file
        h = h5py.File(self.filename_crops)
        #self.crops = h['crops'][:] # LOAD DIRECTLY
        self.crops = h['crops'] # LAZY LOADING
        self.N_crops = self.crops.shape[0]
        self.crop_w = self.crops[0,:].shape[1]
        self.crop_h = self.crops[0,:].shape[2]

        self.grid_width = (self.layout[0] * self.patch_size) + (self.layout[0]-1)*(self.patch_margin - self.patch_overlap_x)
        self.grid_height = (self.layout[1] * self.patch_size) + (self.layout[1]-1) * self.patch_margin
        if self.grid_width > self.crop_w or self.grid_height > self.crop_h:
            print(self.grid_width, self.grid_height)
            print(self.crop_w, self.crop_h)
            raise ValueError('grid too large for image! whoopsie')
        self.grid_offset_x = (self.crop_w - self.grid_width) // 2 # offset to center the grid
        self.grid_offset_y = (self.crop_h - self.grid_height) // 2

        # for debugging print patch extraction locations:
        for y in range(self.layout[1]):
            for x in range(self.layout[0]):
                print(self._compute_position(x,y, force_no_jitter=True), '   ', end='')
            print('')

    def _compute_position(self, idx_x, idx_y, force_no_jitter=False):
        ''' given a valid patch index position compute the pixel position
            for extraction (upper left corner)
            - validates that the position can be cropped
            - also applies jitter (integer amount of random noise on the pixel position)
        '''
        if idx_x >= self.layout[0] or idx_y >= self.layout[1] or idx_x < 0 or idx_y < 0:
            raise ValueError('invalid idx value')

        pos_x = self.grid_offset_x + (self.patch_size + self.patch_margin - self.patch_overlap_x) * idx_x
        pos_y = self.grid_offset_y + (self.patch_size + self.patch_margin) * idx_y

        if pos_x < 0 or pos_x+self.patch_size >= self.crop_w:
            raise ValueError('patch outside of the image - x axis')
        if pos_y < 0 or pos_y+self.patch_size >= self.crop_h:
            raise ValueError('patch outside of the image - y axis')

        if self.jitter is not None and self.jitter > 0 and not force_no_jitter:
            jitter_offset = np.random.randint(low=-self.jitter, high=self.jitter+1, size=2)
            pos_x += jitter_offset[0]
            pos_y += jitter_offset[1]
            # make sure jitter does not break stuff
            if pos_x < 0:
                pos_x = 0
            if pos_x+self.patch_size >= self.crop_w:
                pos_x = self.crop_w-self.patch_size-1
            if pos_y < 0:
                pos_y = 0
            if pos_y+self.patch_size >= self.crop_h:
                pos_y = self.crop_h-self.patch_size-1
        return pos_x, pos_y

    def _random_position(self):
        ''' return (pos_x, pos_y pair) and corresponding label
        '''
        # grid
        idx_x = np.random.randint(self.layout[0])
        idx_y = np.random.randint(self.layout[1])

        return idx_x, idx_y

    def _extract_at(self, crop_idx, position):
        ''' take a crop from the given detection bounding box (stored in self.crops)
            - depending on the objects initialization drop color channel
        '''
        pos_x, pos_y = position
        crop = self.crops[crop_idx, :, pos_x:pos_x+self.patch_size, pos_y:pos_y+self.patch_size]
        return crop

    def _standardize_patch(self, patch):
        if self.patch_mean is not None and self.patch_std is not None: # -- shift and scale
            if len(self.patch_mean.shape) == 1:
                # pixelwise
                mean_bcast = self.patch_mean[np.newaxis, :, np.newaxis, np.newaxis]
                std_bcast = self.patch_std[np.newaxis, :, np.newaxis, np.newaxis]
                patch = ((patch.astype(np.float32) / 255.) - mean_bcast) / std_bcast
            elif len(self.patch_mean.shape) == 3:
                patch = ((patch.astype(np.float32) / 255.) - self.patch_mean) / self.patch_std
        elif self.patch_mean is not None and self.patch_std is None: # -- shift only
            if len(self.patch_mean.shape) == 1:
                # pixelwise
                mean_bcast = self.patch_mean[np.newaxis, :, np.newaxis, np.newaxis]
                patch = ((patch.astype(np.float32) / 255.) - mean_bcast)
            elif len(self.patch_mean.shape) == 3:
                patch = ((patch.astype(np.float32) / 255.) - self.patch_mean)
        else:
            patch = patch.astype(np.float32) / 255.
        if self.range_to_0_255:
            patch = patch * 255.
        return patch

    def __call__(self, crop_idx):
        ''' - decide on random subcrop - take the crop
            - and return the pair (crop, label)
            - !! will return UINT8 --> not scaled to [0,1] !!
        '''
        idx_x, idx_y = self._random_position()
        position = self._compute_position(idx_x, idx_y)
        label = (idx_y * self.layout[0] + idx_x).astype(np.int32)

        crop = self._extract_at(crop_idx, position)
        crop = self._standardize_patch(crop)

        if self.drop_colorchannel:
            # drop one color channel
            chan_idx = np.random.randint(3)
            crop[chan_idx, :] = 0.

        return (crop, label)



class TaskPairwisePatches(TaskPatchesCached):
    ''' cached version of the TaskPatches
        -- here a file with cropped detection bounding boxes is read once
            (created with extract_crops_from_detections.py)

        -- drop_colorchannel : if True, one channel is dropped at random (set to 0)
        -- jitter: None or int > 0, the patch position is jittered by a random amount
    '''
    def __init__(self, filename_crops, patch_size, layout=None, jitter=None, drop_colorchannel=False,
                 patch_mean=None, patch_std=None):
        super(TaskPairwisePatches, self).__init__(filename_crops, patch_size, layout, jitter, drop_colorchannel,
                                                  patch_mean, patch_std)

        self.patches_per_call = 2 # we return 2 patches per call (a 2-tuple)
        N = self.n_patches
        self.label_max = (N-1) * N # we have pairwise label combinations, but not the 'diagonal' pairs (x!=y)

        print('TaskPairwisePatches: returns pair of patches with label in [0..%d]'%self.label_max)

    def _random_position_pair(self):
        all_valid_positions = np.arange(self.layout[0]*self.layout[1])
        np.random.shuffle(all_valid_positions)
        i1 = all_valid_positions[0]
        i2 = all_valid_positions[1]
        N = self.layout[0]
        return [ (i1%N, i1//N), (i2%N, i2//N) ]

    def _index_to_label(self, patch_index):
        idx_x, idx_y = patch_index
        return (idx_y * self.layout[0] + idx_x).astype(np.int32)

    def __call__(self, crop_idx):
        ''' pick 2 random patch positions (non identical!) like the TaskPatchesCached class
            returns both patches and a label, which encodes the patch position relationship
            (label: [0,(N*(N-1)] )
        '''
        indices = self._random_position_pair() # two random positions, which are not identical
        positions = [self._compute_position(i[0], i[1]) for i in indices]

        crop_0 = self._standardize_patch( self._extract_at(crop_idx, positions[0]) )
        crop_1 = self._standardize_patch( self._extract_at(crop_idx, positions[1]) )

        if self.drop_colorchannel:
            # drop one color channel
            chan_idx = np.random.randint(3)
            crop_0[chan_idx, :] = 0.
            chan_idx = np.random.randint(3)
            crop_1[chan_idx, :] = 0.

        label_0 = self._index_to_label(indices[0])
        label_1 = self._index_to_label(indices[1])
        N = self.n_patches
        idx = (label_1 * N) + label_0
        label = idx - ( (idx//(N+1)) + 1 ) # do not enumerate the diagonal entries

        return ((crop_0, crop_1), label)
