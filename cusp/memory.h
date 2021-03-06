 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

/*! \file memory.h
 *  \brief Memory spaces and allocators
 */

#pragma once

#include <cusp/detail/config.h>

#include <thrust/iterator/iterator_traits.h>

namespace cusp
{

#if THRUST_VERSION >= 100600
  typedef thrust::host_system_tag                  host_memory;
  typedef thrust::device_system_tag                device_memory;
  typedef thrust::any_system_tag                   any_memory;
#else
  typedef thrust::host_space_tag                   host_memory;
  typedef thrust::detail::default_device_space_tag device_memory;
  typedef thrust::any_space_tag                    any_memory;
#endif
   
  template<typename T, typename MemorySpace>
  struct default_memory_allocator;
  
  template <typename MemorySpace1, typename MemorySpace2=any_memory, typename MemorySpace3=any_memory>
  struct minimum_space;


} // end namespace cusp

#include <cusp/detail/memory.inl>

