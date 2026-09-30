# import torch

# class EpochUnifiedDataset:
#     def __init__(self, td_offline, td_online):
#         """
#         Treats two independent buffers as a single dataset without copying data.
#         Samples without replacement (proper epochs).
#         """
#         self.td_offline = td_offline
#         self.td_online = td_online
        
#         self.len_off = td_offline.shape[0] if td_offline is not None else 0
#         self.len_on = td_online.shape[0] if td_online is not None else 0
#         self.total_len = self.len_off + self.len_on
        
#         assert self.total_len > 0, "Both offline and online buffers are empty."

#         # 1. Create a fully shuffled list of every index in the dataset
#         self._shuffled_indices = torch.randperm(self.total_len)
#         self._pointer = 0  # Keeps track of where we are in the "epoch"

#     def sample(self, batch_size: int):
#         # 2. Safely grab exactly `batch_size` indices (handles tiny datasets)
#         indices_list = []
#         needed = batch_size
        
#         while needed > 0:
#             available = self.total_len - self._pointer
#             take = min(needed, available)
            
#             indices_list.append(self._shuffled_indices[self._pointer : self._pointer + take])
            
#             self._pointer += take
#             needed -= take
            
#             # If we hit the end of the dataset, reshuffle and reset pointer
#             if self._pointer >= self.total_len:
#                 self._shuffled_indices = torch.randperm(self.total_len)
#                 self._pointer = 0
                
#         indices = torch.cat(indices_list)

#         # 3. Route the indices to the correct underlying TensorDicts
#         off_mask = indices < self.len_off
#         on_mask = ~off_mask
        
#         off_indices = indices[off_mask]
#         batch_off = self.td_offline[off_indices] if len(off_indices) > 0 else None
        
#         on_indices = indices[on_mask] - self.len_off
#         batch_on = self.td_online[on_indices] if len(on_indices) > 0 else None
        
#         # 4. Stitch them together
#         if batch_off is not None and batch_on is not None:
#             combined_batch = torch.cat([batch_off, batch_on], dim=0)
#         elif batch_off is not None:
#             combined_batch = batch_off
#         else:
#             combined_batch = batch_on

#         # 5. Restore the randomization!
#         # torch.cat grouped offline first, then online. We must shuffle the batch itself.
#         return combined_batch[torch.randperm(batch_size)]

#     def __len__(self):
#         return self.total_len


import torch

class EpochUnifiedDataset:
    def __init__(self, td_offline, td_online, sampling_method="fixed_ratio", offline_ratio=0.5):
        """
        Treats two independent buffers as a single dataset without copying data.
        Samples without replacement (proper epochs).
        
        Args:
            sampling_method (str): 
                - "fixed_ratio": Forces a specific percentage of offline vs online per batch.
                - "proportional": Pools everything together, sampling by natural data ratio.
            offline_ratio (float): Fraction of the batch that comes from the offline buffer 
                                   (only used if sampling_method="fixed_ratio"). Default 0.5.
        """
        self.td_offline = td_offline
        self.td_online = td_online
        self.sampling_method = sampling_method
        self.offline_ratio = offline_ratio
        
        # Safely get the lengths of the TensorDicts
        self.len_off = td_offline.shape[0] if td_offline is not None else 0
        self.len_on = td_online.shape[0] if td_online is not None else 0
        self.total_len = self.len_off + self.len_on
        
        assert self.total_len > 0, "Both offline and online buffers are empty."

        if self.sampling_method == "fixed_ratio":
            assert self.len_off > 0 and self.len_on > 0, "Both buffers must have data for fixed_ratio sampling."
            assert 0.0 <= self.offline_ratio <= 1.0, "offline_ratio must be between 0.0 and 1.0"
            
            # Maintain SEPARATE shuffled indices for offline and online
            self._shuffled_off = torch.randperm(self.len_off)
            self._shuffled_on = torch.randperm(self.len_on)
            self._ptr_off = 0
            self._ptr_on = 0
            
        elif self.sampling_method == "proportional":
            # Maintain ONE global shuffled index list
            self._shuffled_indices = torch.randperm(self.total_len)
            self._pointer = 0
            
        else:
            raise ValueError(f"Unknown sampling method: {self.sampling_method}. Use 'fixed_ratio' or 'proportional'.")

    def _get_indices(self, needed, total_len, pointer, shuffled_indices):
        """Helper for fixed_ratio sampling to get `needed` indices from a specific buffer."""
        indices_list = []
        while needed > 0:
            available = total_len - pointer
            take = min(needed, available)
            
            indices_list.append(shuffled_indices[pointer : pointer + take])
            pointer += take
            needed -= take
            
            # Reshuffle if we hit the end of this specific buffer
            if pointer >= total_len:
                shuffled_indices = torch.randperm(total_len)
                pointer = 0
                
        return torch.cat(indices_list), pointer, shuffled_indices

    def sample(self, batch_size: int):
        
        # ==========================================
        # PATH A: The Flexible Fixed-Ratio Approach
        # ==========================================
        if self.sampling_method == "fixed_ratio":
            # Dynamically calculate batch splits based on the ratio
            off_batch = int(batch_size * self.offline_ratio)
            on_batch = batch_size - off_batch # Remainder handles rounding automatically
            
            batch_off, batch_on = None, None
            
            # Pull from offline (only if ratio > 0)
            if off_batch > 0:
                idx_off, self._ptr_off, self._shuffled_off = self._get_indices(
                    off_batch, self.len_off, self._ptr_off, self._shuffled_off
                )
                batch_off = self.td_offline[idx_off]
            
            # Pull from online (only if ratio < 1)
            if on_batch > 0:
                idx_on, self._ptr_on, self._shuffled_on = self._get_indices(
                    on_batch, self.len_on, self._ptr_on, self._shuffled_on
                )
                batch_on = self.td_online[idx_on]
            
            # Combine safely depending on what was actually sampled
            if batch_off is not None and batch_on is not None:
                combined_batch = torch.cat([batch_off, batch_on], dim=0)
            elif batch_off is not None:
                combined_batch = batch_off
            else:
                combined_batch = batch_on
                
            # Shuffle the combined batch itself
            return combined_batch[torch.randperm(batch_size)]
            
        # ==========================================
        # PATH B: The Proportional Approach
        # ==========================================
        elif self.sampling_method == "proportional":
            indices_list = []
            needed = batch_size
            
            while needed > 0:
                available = self.total_len - self._pointer
                take = min(needed, available)
                
                indices_list.append(self._shuffled_indices[self._pointer : self._pointer + take])
                
                self._pointer += take
                needed -= take
                
                # Reshuffle if we hit the end of the global dataset
                if self._pointer >= self.total_len:
                    self._shuffled_indices = torch.randperm(self.total_len)
                    self._pointer = 0
                    
            indices = torch.cat(indices_list)

            # Route the indices to the correct underlying TensorDicts
            off_mask = indices < self.len_off
            on_mask = ~off_mask
            
            off_indices = indices[off_mask]
            batch_off = self.td_offline[off_indices] if len(off_indices) > 0 else None
            
            on_indices = indices[on_mask] - self.len_off
            batch_on = self.td_online[on_indices] if len(on_indices) > 0 else None
            
            # Stitch them together
            if batch_off is not None and batch_on is not None:
                combined_batch = torch.cat([batch_off, batch_on], dim=0)
            elif batch_off is not None:
                combined_batch = batch_off
            else:
                combined_batch = batch_on

            return combined_batch[torch.randperm(batch_size)]

    def __len__(self):
        return self.total_len