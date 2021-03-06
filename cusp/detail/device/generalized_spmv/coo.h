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



#pragma once

#include <cusp/detail/device/spmv/coo_flat.h>

namespace cusp
{
namespace detail
{
namespace device
{

template <typename IndexType, typename ValueType>
void spmv(const coo_matrix<IndexType,ValueType,cusp::device_memory>& coo, 
          const ValueType * x, 
                ValueType * y)
{ 
    spmv_coo_flat(coo, x, y);
}

template <typename IndexType, typename ValueType>
void spmv_tex(const coo_matrix<IndexType,ValueType,cusp::device_memory>& coo, 
              const ValueType * x, 
                    ValueType * y)
{ 
    spmv_coo_flat_tex(coo, x, y);
}

} // end namespace device
} // end namespace detail
} // end namespace cusp

