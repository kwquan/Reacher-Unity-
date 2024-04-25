using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Rotation : MonoBehaviour
{
    private GameObject joint;
    private float xAngle = 0;
    private float yAngle = 0.8f;
    private float zAngle = 0;
    // Start is called before the first frame update
    void Start()
    {
        joint = GameObject.Find("Joint");
    }

    // Update is called once per frame
    void Update()
    {
        joint.transform.Rotate(xAngle, yAngle, zAngle);
    }
}
