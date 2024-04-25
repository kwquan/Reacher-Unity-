using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.SceneManagement;
using UnityEngine.Networking;
using UnityEditor;
using System;

public class Env : MonoBehaviour
{
    // Start is called before the first frame update
    private GameObject joint;
    private GameObject joint2;
    string state;
    string experience;
    public static Action action = new Action();
    public static BufferLength buffer_length = new BufferLength();
    void Start()
    {
        StartCoroutine("check");
        joint = GameObject.Find("Joint");
        joint2 = GameObject.Find("Joint2");
        state = GetState();
        //print(Vector3.Distance(GameObject.Find("Point").transform.position, GameObject.Find("Target").transform.position));
    }
    int steps = 0;
    int max_steps = 50;
    static int episodes = 0;
    int min_memory_len = 500;
    double episode_reward = 0;
    static List<double> episode_rewards = new List<double>();
    static List<double> early_term_rewards = new List<double>();
    int terminate = 0;
    int episode_end = 0;
    // Update is called once per frame
    void Update()
    {
        steps += 1;
        StartCoroutine(get_action((i) => { action.action_0 = i.action_0; action.action_1 = i.action_1; }));
        joint.transform.Rotate(0, action.action_0, 0);
        joint2.transform.Rotate(0, action.action_1, 0);
        double reward = GetReward(action.action_0, action.action_1);
        terminate = GetTerminate(steps, reward);
        string next_state = GetState();
        experience = GetExp(state, action, reward, terminate, next_state);
        StartCoroutine(get_experience((i) => { buffer_length.length = i.length; print(i.length); }));
        episode_end = GetEpisodeEnd();
        state = next_state;
        episode_reward += reward;
        System.Threading.Thread.Sleep(100);
        print(steps);
        if (episode_end == 1)
        {
            Reset();
            print("episodes ended: " + episodes);
            print("buffer length: " + buffer_length.length);
            if (buffer_length.length >= min_memory_len)
            {
                StartCoroutine(update_weights());
                if (episodes % 5 == 0)
                {
                    StartCoroutine(update_target_weights());
                }
                CheckSolve();
            }
        }
    }

    string GetState()
    {
        float joint2_hyp = Vector3.Distance(GameObject.Find("Joint").transform.position, GameObject.Find("Joint2").transform.position);
        float joint2_x = GameObject.Find("Joint2").transform.position.x;
        float joint2_z = GameObject.Find("Joint2").transform.position.z;
        float joint_sine = joint2_z / joint2_hyp;
        float joint_cosine = joint2_x / joint2_hyp;
        float point_hyp = Vector3.Distance(GameObject.Find("Joint2").transform.position, GameObject.Find("Point").transform.position);
        float point_x = GameObject.Find("Point").transform.position.x - joint2_x;
        float point_z = GameObject.Find("Point").transform.position.z - joint2_z;
        float point_sine = point_z / joint2_hyp;
        float point_cosine = point_x / joint2_hyp;
        float target_x = GameObject.Find("Target").transform.position.x;
        float target_z = GameObject.Find("Target").transform.position.z;
        float section_ang_speed = GameObject.Find("Section").GetComponent<Rigidbody>().angularVelocity.magnitude;
        float section_2_ang_speed = GameObject.Find("Section2").GetComponent<Rigidbody>().angularVelocity.magnitude;
        float distance_x = point_x - target_x;
        float distance_z = point_z - target_z;

        string state_tuple = "{\"joint_sine\":\"" + joint_sine + "\", \"joint_cosine\":\"" + joint_cosine + "\", \"point_sine\":\"" + point_sine + "\", \"point_cosine\":\"" + point_cosine + "\", \"section_ang_speed\":\"" + section_ang_speed + "\", \"section_2_ang_speed\":\"" + section_2_ang_speed + "\", \"distance_x\":\"" + distance_x + "\", \"distance_z\":\"" + distance_z + "\"}";
        return state_tuple;
    }

    string GetExp(string state, Action action, double reward, int terminate, string next_state)
    {
        string exp_action = "{\"action_0\":\"" + action.action_0 + "\", \"action_1\":\"" + action.action_1 + "\"}";
        string exp_reward = "{\"reward\":\"" + reward + "\"}";
        string exp_terminate = "{\"terminate\":\"" + terminate + "\"}";
        string experience_tuple = "[" + state + "," + exp_action + "," + exp_reward + "," + exp_terminate + "," + next_state + "]";
        return experience_tuple;
    }

    double GetReward(float action_0, float action_1)
    {
        double reward_euclidean = Vector3.Distance(GameObject.Find("Point").transform.position, GameObject.Find("Target").transform.position);
        double reward_smooth = (Math.Abs(action_0) + Math.Abs(action_1))/12;
        double reward_total = -(reward_euclidean + reward_smooth);
        //print("reward_euclidean: " + reward_euclidean);
        //print("reward_smooth: " + reward_smooth);
        print("total reward: " + reward_total);
        return reward_total;
    }

    int CheckReached()
    {
        double reward_euclidean = Vector3.Distance(GameObject.Find("Point").transform.position, GameObject.Find("Target").transform.position);
        if (reward_euclidean < 0.1)
        {
            return 1;
        }
        return 0;
    }

    int GetTerminate(int steps, double reward)
    {
        int check_reached = CheckReached();
        if(check_reached == 1)
        {
            early_term_rewards.Add(episode_reward);
            print("episode reward[solved]: " + episode_reward);
            return 1;
        }
        return 0;
    }

    int GetEpisodeEnd()
    {
        if (steps == max_steps)
        {
            return 1;
        }
        return 0;
    }

    void Reset()
    {
        episodes += 1;
        episode_rewards.Add(episode_reward);
        episode_reward = 0;
        steps = 0;
        episode_end = 0;
        //print("episode " + episodes + " finished");
        SceneManager.LoadScene(SceneManager.GetActiveScene().name);
        terminate = 0;
        foreach (double item in episode_rewards)
        {
            print(item);
        }
        System.Threading.Thread.Sleep(2000);
    }

    void CheckSolve()
    {
        double sum = 0;
        var rewards = episode_rewards.ToArray();
        Array.Reverse(rewards);
        Array.Resize(ref rewards, 5);
        Array.ForEach(rewards, x => sum += x);
        if(sum/5 > -30)
        {
            print("env solved!");
            print("average reward: " + Math.Round(sum / 5,1));
            foreach (double item in early_term_rewards)
            {
                print(item);
            }
            StartCoroutine(save_weights());
            EditorApplication.isPlaying = false;
        }
    } 

    IEnumerator check()
    {
        using (UnityWebRequest webRequest = UnityWebRequest.Get("http://127.0.0.1:5000/check"))
        {
            yield return webRequest.SendWebRequest();
            if(webRequest.result == UnityWebRequest.Result.ConnectionError)
            {
                Debug.Log("Error: " + webRequest.error);
            } else
            {
                Debug.Log(webRequest.downloadHandler.text);
            }
        }
    }

    IEnumerator get_action(System.Action<Action> callback)
    {
        using (UnityWebRequest webRequest = UnityWebRequest.Post("http://127.0.0.1:5000/get_action", state, "application/json"))
        {
            yield return webRequest.SendWebRequest();
            if (webRequest.result == UnityWebRequest.Result.ConnectionError)
            {
                Debug.Log("Error: " + webRequest.error);
            }
            else
            {
                Action a = JsonUtility.FromJson<Action>(webRequest.downloadHandler.text);
                callback(a);
            }
        }

    }

    IEnumerator get_experience(System.Action<BufferLength> callback)
    {
        using (UnityWebRequest webRequest = UnityWebRequest.Post("http://127.0.0.1:5000/get_experience", experience, "application/json"))
        {
            yield return webRequest.SendWebRequest();
            if (webRequest.result == UnityWebRequest.Result.ConnectionError)
            {
                Debug.Log("Error: " + webRequest.error);
            }
            else
            {
                BufferLength b = JsonUtility.FromJson<BufferLength>(webRequest.downloadHandler.text);
                callback(b);
            }
        }

    }

    IEnumerator update_weights()
    {
        using (UnityWebRequest webRequest = UnityWebRequest.Get("http://127.0.0.1:5000/update_weights"))
        {
            
            yield return webRequest.SendWebRequest();
            if (webRequest.result == UnityWebRequest.Result.ConnectionError)
            {
                Debug.Log("Error: " + webRequest.error);
            }
            else
            {
                Debug.Log(webRequest.downloadHandler.text);
            }
        }
    }

    IEnumerator update_target_weights()
    {
        using (UnityWebRequest webRequest = UnityWebRequest.Get("http://127.0.0.1:5000/update_target_weights"))
        {

            yield return webRequest.SendWebRequest();
            if (webRequest.result == UnityWebRequest.Result.ConnectionError)
            {
                Debug.Log("Error: " + webRequest.error);
            }
            else
            {
                Debug.Log(webRequest.downloadHandler.text);
            }
        }
    }

    IEnumerator save_weights()
    {
        using (UnityWebRequest webRequest = UnityWebRequest.Get("http://127.0.0.1:5000/save_weights"))
        {

            yield return webRequest.SendWebRequest();
            if (webRequest.result == UnityWebRequest.Result.ConnectionError)
            {
                Debug.Log("Error: " + webRequest.error);
            }
            else
            {
                Debug.Log(webRequest.downloadHandler.text);
            }
        }
    }

    // Action class for action interpretation 
    [System.Serializable]
    public class Action
    {
        public float action_0;
        public float action_1;
    }

    [System.Serializable]
    public class BufferLength
    {
        public int length;
    }
}
