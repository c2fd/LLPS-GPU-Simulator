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

/*! \file version.h
 *  \brief Cusp version
 */

#pragma once

#include <cusp/detail/config.h>

//  This is the only cusp header that is guaranteed to 
//  change with every cusp release.
//
//  CUSP_VERSION % 100 is the sub-minor version
//  CUSP_VERSION / 100 % 1000 is the minor version
//  CUSP_VERSION / 100000 is the major version

#define CUSP_VERSION 400
#define CUSP_MAJOR_VERSION     (CUSP_VERSION / 100000)
#define CUSP_MINOR_VERSION     (CUSP_VERSION / 100 % 1000)
#define CUSP_SUBMINOR_VERSION  (CUSP_VERSION % 100)

