package hw3;

import java.io.InputStream;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.ListIterator;
import java.util.Map;
import java.util.Random;
import java.util.Set;

import edu.bu.hw3.linalg.Matrix;
import edu.bu.hw3.nn.LossFunction;
import edu.bu.hw3.nn.Model;
import edu.bu.hw3.nn.Optimizer;
import edu.bu.hw3.nn.layers.Dense;
import edu.bu.hw3.nn.layers.ReLU;
import edu.bu.hw3.nn.layers.Sigmoid;
import edu.bu.hw3.nn.layers.Tanh;
import edu.bu.hw3.nn.losses.MeanSquaredError;
import edu.bu.hw3.nn.models.Sequential;
import edu.bu.hw3.nn.optimizers.SGDOptimizer;
import edu.bu.hw3.streaming.Streamer;
import edu.bu.hw3.utils.Pair;
import edu.bu.hw3.utils.Triple;
import edu.cwru.sepia.action.Action;
import edu.cwru.sepia.action.ActionFeedback;
import edu.cwru.sepia.action.ActionResult;
import edu.cwru.sepia.action.ActionType;
import edu.cwru.sepia.agent.Agent;
import edu.cwru.sepia.environment.model.history.DamageLog;
import edu.cwru.sepia.environment.model.history.DeathLog;
import edu.cwru.sepia.environment.model.history.History.HistoryView;
import edu.cwru.sepia.environment.model.state.Unit;
import edu.cwru.sepia.environment.model.state.Unit.UnitView;
import edu.cwru.sepia.environment.model.state.UnitTemplate.UnitTemplateView;
import edu.cwru.sepia.util.Direction;
import hw2.chess.game.move.Move;
import hw2.chess.utils.Coordinate;
import edu.cwru.sepia.environment.model.state.State.StateView;
import edu.cwru.sepia.environment.model.state.Template.TemplateView;

public class QAgent extends Agent
{

	public static final long serialVersionUID = -5077535504876086643L;
	public static final int RANDOM_SEED = 12345;
	public static final double GAMMA = 0.9; //0.9
	public static final double LEARNING_RATE = 0.00001; //0.0001
	public static final double EPSILON = 0.02; // 0.02 prob of ignoring the policy and choosing a random action

	// our agent will play this many training episodes in a row before testing
	public static final int NUM_TRAINING_EPISODES_IN_BATCH = 10;
                                 
	// our agent will play this many testing episodes in a row before training again
	public static final int NUM_TESTING_EPISODES_IN_BATCH = 5;

	private final String paramFilePath;

	private Streamer streamer;

	private final int NUM_EPISODES_TO_PLAY;

	private int numTestEpisodesPlayedInBatch = -1;
	private int numTrainingEpisodesPlayed = 0;

	// rng to keep things repeatable...will combine with the RANDOM_SEED
	public final Random random;

	private Integer ENEMY_PLAYER_ID; // initially null until initialStep() is called

	private Set<Integer> myUnits;
	private Set<Integer> enemyUnits;
	private List<Double> totalRewards;

	/** NN specific things **/
	private Model qFunctionNN;
	private LossFunction lossFunction;
	private Optimizer optimizer;

	// how we remember what was the state, Q-value, and reward from the past
	private Map<Integer, Triple<Matrix, Matrix, Double> > oldInfoPerUnit;
	
	private int initialMyUnits = 0;
	private int initialEnemyUnits = 0;
	public QAgent(int playerId, String[] args)
	{
		super(playerId);
		String streamerArgString = null;
		String paramFilePath = null;

		if(args.length < 3)
		{
			System.err.println("QAgent.QAgent [ERROR]: need to specify playerId, streamerArgString, paramFilePath");
			System.exit(-1);
		}

		streamerArgString = args[1];
		paramFilePath = args[2];

		int numEpisodesToPlay = QAgent.NUM_TRAINING_EPISODES_IN_BATCH; // train in total
		boolean loadParams = false;
		if(args.length >= 4)
		{
			numEpisodesToPlay = Integer.parseInt(args[3]); //if specify then update
			if(args.length >= 5)
			{
				loadParams = Boolean.parseBoolean(args[4]); // if specify then update
			}
		}

		this.NUM_EPISODES_TO_PLAY = numEpisodesToPlay;
		this.ENEMY_PLAYER_ID = null; // initially

		this.paramFilePath = paramFilePath;

		this.myUnits = null;
		this.enemyUnits = null;
		this.totalRewards = new ArrayList<Double>((int)this.NUM_EPISODES_TO_PLAY / QAgent.NUM_TRAINING_EPISODES_IN_BATCH);
		this.totalRewards.add(0.0); // ?

		this.streamer = Streamer.makeDefaultStreamer(streamerArgString, this.getPlayerNumber());
		this.random = new Random(QAgent.RANDOM_SEED);
		
		this.qFunctionNN = this.initializeQFunction(loadParams);
		this.lossFunction = new MeanSquaredError();
		this.optimizer = new SGDOptimizer(this.getQFunction().getParameters(),
				QAgent.LEARNING_RATE);
		this.oldInfoPerUnit = new HashMap<Integer, Triple<Matrix, Matrix, Double> >(); // initialize(?)
		this.initialMyUnits = 0;
		this.initialEnemyUnits = 0;
		
	}

	private final String getParamFilePath() { return this.paramFilePath; }
	private Integer getEnemyPlayerId() { return this.ENEMY_PLAYER_ID; }
	private Set<Integer> getMyUnitIds() { return this.myUnits; }
	private Set<Integer> getEnemyUnitIds() { return this.enemyUnits; }
	private List<Double> getTotalRewards() { return this.totalRewards; }
	private final Streamer getStreamer() { return this.streamer; }
	private final Random getRandom() { return this.random; }

	/** NN specific stuff **/
	private Model getQFunction() { return this.qFunctionNN; }
	private LossFunction getLossFunction() { return this.lossFunction; }
	private Optimizer getOptimizer() { return this.optimizer; }
	private Map<Integer, Triple<Matrix, Matrix, Double> > getOldInfoPerUnit() { return this.oldInfoPerUnit; }

	private boolean isTrainingEpisode() { return this.numTestEpisodesPlayedInBatch == -1; }

	/**
	 * A method to create the neural network used for the Q function.
	 * You can make it as deep as you want to (although it will take more time to compute)
	 * 
	 * The API for creating a neural network is as follows:
	 *     Sequential m = new Sequential();
	 *     // layer 1
	 *     m.add(new Dense(feature_dim, hidden_dim1, this.getRandom()));
	 *     m.add(Sigmoid());
	 *     
	 *     // layer 2
	 *     m.add(new Dense(hidden_dim1, hidden_dim2, this.getRandom()));
	 *     m.add(Tanh());
	 *     
	 *     // add as many layers as you want
	 *     
	 *     // the last layer MUST be a scalar though
	 *     m.add(new Dense(hidden_dimN, 1));
	 *     m.add(ReLU()); // decide if you want to add an activation
	 * 
	 * @param loadParams
	 * @return
	 */
	private Model initializeQFunction(boolean loadParams)
	{
		Sequential m = new Sequential();

		/**
		 * TODO: create your model!
		 */
		int feature_dim = 14;
		int hidden_dim1 = 256;
		int hidden_dim2 = 256;
		int hidden_dim3 = 512;
		int hidden_dim4 = 256;
		int hidden_dim5 = 128;
		int hidden_dim6 = 64;
		m.add(new Dense(feature_dim, hidden_dim1, this.getRandom()));
		m.add(new Sigmoid());
		m.add(new Dense(hidden_dim1, hidden_dim2, this.getRandom()));
		m.add(new Sigmoid());
		m.add(new Dense(hidden_dim2, hidden_dim3));
		m.add(new ReLU());
		m.add(new Dense(hidden_dim3, hidden_dim4));
		m.add(new Tanh());
		m.add(new Dense(hidden_dim4, hidden_dim5));
		m.add(new Sigmoid());
		m.add(new Dense(hidden_dim5, 1));
		m.add(new Sigmoid());
		if(loadParams)
		{
			try
			{
				m.load(this.getParamFilePath());
			} catch (Exception e)
			{
				// TODO Auto-generated catch block
				e.printStackTrace();
				System.exit(-1);
			}
		}
		return m;
	}

	/**
     * Given the current state and the footman in question calculate the reward received on the last turn.
     * This is where you will check for things like Did this footman take or give damage? Did this footman die
     * or kill its enemy. Did this footman start an action on the last turn? 
     *
     * Remember that you will need to discount this reward based on the timestep it is received on.
     *
     * As part of the reward you will need to calculate if any of the units have taken damage. You can use
     * the history view to get a list of damages dealt in the previous turn. Use something like the following.
     *
     * for(DamageLog damageLogs : historyView.getDamageLogs(lastTurnNumber)) {
     *     System.out.println("Defending player: " + damageLog.getDefenderController() + " defending unit: " + \
     *         damageLog.getDefenderID() + " attacking player: " + damageLog.getAttackerController() + \
     *         "attacking unit: " + damageLog.getAttackerID());
     * }
     *
     * You will do something similar for the deaths. See the middle step documentation for a snippet
     * showing how to use the deathLogs.
     *
     * To see if a command was issued you can check the commands issued log.
     *
     * Map<Integer, Action> commandsIssued = historyView.getCommandsIssued(playernum, lastTurnNumber);
     * for (Map.Entry<Integer, Action> commandEntry : commandsIssued.entrySet()) {
     *     System.out.println("Unit " + commandEntry.getKey() + " was command to " + commandEntry.getValue().toString);
     * }
     *
     * @param state The current state of the game.
     * @param history History of the episode up until this turn.
     * @param unitId The id of the unit you are looking to calculate the reward for.
     * @return The current reward for that unit
     */
	private double getRewardForUnit(StateView state, HistoryView history, int unitId)
    {
     /** TODO: complete me! **/
       double reward = 0.0;
       int lastTurnNumber = state.getTurnNumber() - 1;
    // record the defender unit and num they are attacked
       Map<Integer,Integer> defendingEnemyUnit = new HashMap<>(); 
       Integer unitIamAttacking = null;
       boolean damageLogEmpty = true;
       
       for(DamageLog damageLogs : history.getDamageLogs(lastTurnNumber)) {
           // if i am attacking, reward. else, reward reduce
    	   damageLogEmpty = false;
    	   int defenderID = damageLogs.getDefenderID();
    	   int attackerID = damageLogs.getAttackerID();
    	   
    	   if(this.getMyUnitIds().contains(attackerID)) { // if we are attack enemy
    		   int num = defendingEnemyUnit.getOrDefault((Integer)defenderID, (Integer)0);
    		   defendingEnemyUnit.put(defenderID,num+1);
        	   if (attackerID == unitId) {
        		   reward += damageLogs.getDamage()*1.0; // if I attack reward
        		   unitIamAttacking = defenderID;
        		   //System.out.println("me attack: "+reward);
        	   }
    	   }else {// if my unit are being attacked
        	   if(damageLogs.getDefenderID() == unitId){
        		   //reduce reward according to my health, the lower my health,  higher the penalty
        		   UnitView thisUnit = state.getUnit(unitId);
        		   double myHealthPercent = ((double)thisUnit.getHP())/((double)thisUnit.getTemplateView().getBaseHealth());
        		   reward -= damageLogs.getDamage()*(1-myHealthPercent)*1.5; 
        		   //System.out.println("I was attacked: "+reward);
        	   }
    	   }
       }
       
       if(unitIamAttacking != null) {// if I attack
    	   double unitIAttackSum = defendingEnemyUnit.getOrDefault(unitIamAttacking,1);
    	   if(unitIAttackSum > 1) { // and the defender is the one my teamates also attack
    		   reward += unitIAttackSum*1.0; // reward gang attack
    	   }
    	   double unitIAttackHealthPercent = 0.0;
    	   if(state.getUnit(unitIamAttacking) != null) {
        	   unitIAttackHealthPercent = (double)state.getUnit(unitIamAttacking).getHP()/(double)state.getUnit(unitIamAttacking).getTemplateView().getBaseHealth();
    	   }
    	   if(unitIAttackHealthPercent<0.5) { // if the unit I attack is at low health, reward
    		   reward += (1.0-unitIAttackHealthPercent)*10.0;
    	   }
    	   
       }
       
       for(DeathLog deathLogs : history.getDeathLogs(lastTurnNumber)) {
    	   Integer deadId = deathLogs.getDeadUnitID();
    	   //System.out.println(deadId+ " "+unitIamAttacking);
    	   boolean s1 = unitIamAttacking !=null;
    	   boolean s2 = (deadId == unitIamAttacking);
    	   //System.out.println(s1 + " "+s2);
    	   if (deathLogs.getController() == this.getPlayerNumber()) {// DEAD CODE: if I die, penalty
        	   reward -= 100.0;
    		   //System.out.println("I die");
           }else if(s1 && s2) {
        	   // System.out.println("enemy die: "+reward);
        	   reward += 800.0; // reward if I kill a unitÓÓ˜ 
        	   //System.out.println("I kill");
           }
           //System.out.println("Player: " + deathLogs.getController() + " unit: " + deathLogs.getDeadUnitID());
       }
      
       
       UnitView unit = state.getUnit(unitId);
       double enemyNeiNum = 0.0;
       double friendNeiNum = 0.0;
       int unitXpos = unit.getXPosition();
       int unitYpos = unit.getYPosition();
       //System.out.println("X:"+unitXpos+" Y: "+unitYpos);
       for (int i = unitXpos - 1; i <= unitXpos + 1; i++) { // run from pt.x-1 -> pt.x+1
    	   for (int j = unitYpos - 1; j <= unitYpos + 1; j++) { // run from pt.y-1 -> pt.y+1
    		   if (i != unitXpos || j !=unitYpos) { // if we arent at (pt.x, pt.y)
    			  // if the point is valid (not occupied and in range)
    			  if (state.inBounds(i, j)) {
    				  if(state.isUnitAt(i, j)) {
    					  int neigid = state.unitAt(i, j);
    					  if(this.getMyUnitIds().contains(neigid)) {
    						  friendNeiNum += 1;
    					  }else {
    						  enemyNeiNum += 1; // there is at least one enemy unit in neighbor
    						  //System.out.println("it is my enemy so add reward");
    					  }
    				  }
    			  }
    		  }
    	  }
      }
       if(enemyNeiNum > 0) {
    	   if(enemyNeiNum == 1 || (enemyNeiNum>=(friendNeiNum+1))) {
    		   reward += 0.2;
    		   //System.out.println("more friend: "+reward);
    	   }else {
    		   reward -= (enemyNeiNum-friendNeiNum)*0.1;
    		   //System.out.println("more enemy: "+reward);
    	   }
       }else {// if no enemy around
    	   //if no damage attack or no one die then assume enemy and friend has not met
    	   if(this.myUnits.size() != this.initialMyUnits || !damageLogEmpty) {
    		   reward -= .5;
    		   //System.out.println("no enemy: "+reward);
    	   }
    	   
       }
//       double myHP=0.0;
//       
//       for(Integer unitId1: state.getUnitIds(this.getPlayerNumber()))
//       {
//        UnitView unitView = state.getUnit(unitId1);
//        myHP+=unitView.getHP();
//       
//       }
//       System.out.println("my health: "+myHP);
//       
//       double enemyHP=0.0;
//       for(Integer unitId2: state.getUnitIds(this.getEnemyPlayerId()))
//       {
//        UnitView unitView = state.getUnit(unitId2);
//        enemyHP+=unitView.getHP();
//        
//       }
//       System.out.println("enemy health: "+enemyHP);
       //System.out.println("final reward: "+ reward);
       return reward;
    }

    /**
    * Given a state and action calculate your features here. Please include a comment explaining what features
    * you chose and why you chose them.
    *
    * All of your feature functions should evaluate to a double. Collect all of these into a row vector
    * (a Matrix with 1 row and n columns). This will be the input to your neural network
    *
    * It is a good idea to make the first value in your array a constant. This just helps remove any offset
    * from 0 in the Q-function. The other features are up to you.
    * 
    * It might be a good idea to save whatever feature vector you calculate in the oldFeatureVectors field
    * so that when that action ends (and we observe a transition to a new state), we can update the Q value Q(s,a)
    *
    * @param state Current state of the SEPIA game
    * @param history History of the game up until this turn
    * @param atkUnitId Your unit. The one doing the attacking.
    * @param tgtUnitId An enemy unit. The one your unit is considering attacking.
    * @return The Matrix of feature function outputs.
    */
	private Matrix calculateFeatureVector(StateView state, HistoryView history,
	        int atkUnitId, int tgtUnitId) {
	       // Initialize a feature vector of size n, where n is the number of features
	     int n = 14;
	     // make the first value in your array a constant
	     Matrix featureVector =Matrix.zeros(1, n);
	     //System.out.println("feature vector is:"+featureVector);
	     
	     // Set the first value to 1.0 as a constant feature
	     featureVector.set(0, 0, 1.0);
	     
	     double XExtent = state.getXExtent();
	     double YExtent = state.getYExtent();
	     	     
	     // Get the attacking and target units
	     UnitView atkUnit = state.getUnit(atkUnitId);
	     UnitView tgtUnit = state.getUnit(tgtUnitId);
	     
	     UnitTemplateView atkTemplate = atkUnit.getTemplateView();
	     UnitTemplateView tgtTemplate = tgtUnit.getTemplateView();
	     
	     double atkX = atkUnit.getXPosition();
	     double tgtX = tgtUnit.getXPosition();
	     double atkY = atkUnit.getYPosition();
	     double tgtY = tgtUnit.getYPosition();
	     // location input for calculation of distance and etc
	     double atkxpostion = atkX/XExtent;
 	     double tgtxpostion = tgtX/XExtent;
	     double atkypostion = atkY/YExtent;
	     double tgtypostion = tgtY/YExtent;
	     
	     
	     //health percentage for each unit
	     double atkHP = atkUnit.getHP()/atkTemplate.getBaseHealth();
	     double tgtHp = tgtUnit.getHP()/tgtTemplate.getBaseHealth();
	     
	     //input basic attack for each unit
	     double atkAttack = atkTemplate.getBasicAttack();
	     double tgtAttack = tgtTemplate.getBasicAttack();
	     double atckRatio = atkAttack/tgtUnit.getHP();
	       
	     //calculate proportion of our surviving footman 
	     double EnemiesPct = (this.getEnemyUnitIds().size())/this.initialEnemyUnits;
	     double FriendPct = (this.getMyUnitIds().size())/this.initialMyUnits;
	     //double footmanNumDiff = totalFriends - totalEnemies;
        
	     // Calculate the distance between the attacking and target units, and using the length of diagonal to make it relative
	     double wholeMapDist = Math.sqrt(Math.pow(XExtent,2)+Math.pow(YExtent,2));
	     double distance = Math.sqrt(Math.pow(atkX - tgtX, 2) + Math.pow(atkX - tgtY, 2));
	     distance = distance/wholeMapDist;
        
	     // build map for faster lookup of enemyunit
	     Map<Integer, Double> enemyXmap = new HashMap<>(); // hashmap to find enemy x axis
	     Map<Integer, Double> enemyYmap = new HashMap<>(); // like above for y
	     for(Integer enemyUnitId :this.getEnemyUnitIds()) {
	    	 UnitView enemyUnit = state.getUnit(enemyUnitId);
	    	 double enemyX = enemyUnit.getXPosition();
	    	 double enemyY = enemyUnit.getYPosition();
	    	 enemyXmap.put(enemyUnitId, enemyX);
	    	 enemyYmap.put(enemyUnitId, enemyY);
	     }
        
	     Map<Integer, ActionResult> prevUnitActionsResult = history.getCommandFeedback(this.getPlayerNumber(), state.getTurnNumber() - 1);
        
	     Map<Integer, ActionResult> EnemyprevUnitActionsResult = history.getCommandFeedback(this.getEnemyPlayerId(), state.getTurnNumber() - 1);

	     //calculate the distance between every single one of our footman to all the other enemies.
	     double totalDistance=0.0;
	     // how many teammates are attacking
	     double attackFriends = 0.0;
	     double attackEnemy = 0.0;
	     for (Integer myUnitId :this.getMyUnitIds()) {
	    	 UnitView myUnit = state.getUnit(myUnitId);
	    	 double myX = myUnit.getXPosition();
	    	 double myY = myUnit.getYPosition();
	    	 
	    	 if(!prevUnitActionsResult.isEmpty() && prevUnitActionsResult.containsKey(myUnitId)) {
	    		 ActionResult actionRes = prevUnitActionsResult.get(myUnitId);
	    		 if(prevUnitActionsResult.get(myUnitId).getFeedback().equals(ActionFeedback.INCOMPLETE) &&
	    				 actionRes.getAction().getType().equals(ActionType.COMPOUNDATTACK)) {
		    		 attackFriends += 1;
		    	 }
	    	 }
	    	 for (Integer enemyUnitId :this.getEnemyUnitIds()) {
	    		 double enemyX = enemyXmap.get(enemyUnitId);
	             double enemyY = enemyYmap.get(enemyUnitId);
	             totalDistance += Math.sqrt(Math.pow(myX - enemyX, 2) + Math.pow(myY - enemyY, 2));
	             if(!EnemyprevUnitActionsResult.isEmpty() && EnemyprevUnitActionsResult.containsKey(enemyUnitId)) {
	            	 ActionResult actionRes = EnemyprevUnitActionsResult.get(enemyUnitId);
	            	 if(actionRes.getFeedback().equals(ActionFeedback.INCOMPLETE) &&
	            			 actionRes.getAction().getType().equals(ActionType.COMPOUNDATTACK)) {
			        	 attackEnemy += 1;
			         }
	             }
		         
	    	 }
	     }
	     // use length of diagonal to make it relative
	     totalDistance = totalDistance/((this.getMyUnitIds().size())*(this.getEnemyUnitIds().size())*wholeMapDist);
	     
	     attackFriends = attackFriends/(this.getMyUnitIds().size());
	     attackEnemy = attackEnemy/(this.getEnemyUnitIds().size());
	     
	     Triple<Matrix, Matrix, Double> prev_info = this.getOldInfoPerUnit().get(atkUnitId);
	     double prev_atkHP = 1.0;
	     double prev_tgtHp = 1.0;
	     if(prev_info != null) {
		     Matrix prev_featureVec = prev_info.getFirst();
		     prev_atkHP = prev_featureVec.get(0, 1);
		     prev_tgtHp = prev_featureVec.get(0, 2);
	     }

//////////////////////////////////////////////////////////////////////////////////////////
	  // Set the feature values in the feature vector
	     
	  // postion as input because maybe a particular position is better
	  // the position are relative to X-Extend and Y-Extend
	    featureVector.set(0, 1, atkxpostion); 
	    featureVector.set(0, 2, tgtxpostion);
	    featureVector.set(0, 3, atkypostion);
	    featureVector.set(0, 4, tgtypostion);
	    
	  //myunit health percentage wrt. basic health, can decide whether to run away or attack
	    featureVector.set(0, 5, atkHP); 
	  //enemy health percentage, if low then we may be able to kill, otherwise we can choose other target
	    featureVector.set(0, 6, tgtHp); 
	   
	 // atckRio compare my basic attack to enemy's health
	 //  if my attack damage is higher then I may prefer attacking as I have higher chance to kill
	    featureVector.set(0, 7, atckRatio); 
	    
	    // distance feature: if attack unit is far from target, might want to choose closer target
	    featureVector.set(0, 8, distance); // also make it percentage to the diagonal of whole map
	    // whole distance: if my units are generally far away from enemy, I might want to get closer
	    featureVector.set(0, 9, totalDistance);
	    
	    // percentage of my unit that are attacking: might prefer gang attack
	    featureVector.set(0, 10, attackFriends); 
	    // percentage of enemy unit that are attacking: might want to avoid
	    featureVector.set(0, 11, attackEnemy);
	    
	    // surviving unit in enemy teams and my teams
	    featureVector.set(0, 12, EnemiesPct);
	    featureVector.set(0, 13, FriendPct);
//	    // Previous atcking unit HP and tgtHP
//	    featureVector.set(0, 14, prev_atkHP);
//	    featureVector.set(0, 15, prev_tgtHp);
	    
	    //System.out.println("feature vector is:"+featureVector);
	    return featureVector;
	    }
   
    /**
     * Calculate the Q-Value for a given state action pair. The state in this scenario is the current
     * state view and the history of this episode. The action is the attacker and the enemy pair for the
     * SEPIA attack action.
     *
     * This returns the Q-value according to your feature approximation. This is where you will pass
     * your features through your network (and extract the predicted q-value using the .item() method)
     * @param featureVec The feature vector
     * @return The approximate Q-value
     */
    private double calculateQValue(Matrix featureVec)
    {
    	double qValue = 0.0;
        try
        {
			qValue = this.getQFunction().forward(featureVec).item();
		} catch (Exception e)
        {
			System.err.println("QAgent.caculateQValue [ERROR]: error in either forward() or item()");
			e.printStackTrace();
			System.exit(-1);
		}
        return qValue;
    }

    /**
     * Given a unit and the current state and history of the game select the enemy that this unit should
     * attack. This is where you would do the epsilon-greedy action selection.
     * 
     * You will need to consider who to attack. A unit should always be attacking
     * (if it is not currently attacking something), so what makes actions "different"
     * is who the unit is attacking
     *
     * @param state Current state of the game
     * @param history The entire history of this episode
     * @param atkUnitId The unit (your unit) that will be attacking
     * @return The enemy footman ID this unit should attack
     */
    private int selectAction(StateView state, HistoryView history, int atkUnitId)
    {
    	Integer tgtUnitId = null;
    	Matrix featureVec = null;
    	double maxQ = Double.NEGATIVE_INFINITY;
    	double r = this.getRewardForUnit(state, history, atkUnitId);

    	// epsilon-greedy (i.e. random exploration function)
    	if(this.getRandom().nextDouble() < QAgent.EPSILON && this.isTrainingEpisode()) //&& this.isTrainingEpisode()
    	{
    		// ignore policy and choose a random action (i.e. attacking which enemy)
    		int randomEnemyIdx = this.getRandom().nextInt(this.getEnemyUnitIds().size());

    		// get the unitId at that position
    		tgtUnitId = this.getEnemyUnitIds().toArray(new Integer[this.getEnemyUnitIds().size()])[randomEnemyIdx];
    		featureVec = this.calculateFeatureVector(state, history, atkUnitId, tgtUnitId);
    		maxQ = this.calculateQValue(featureVec);
    	} else
    	{
	    	// find the action (i.e. attacking which enemy) that maximizes the Q-value
	    	for(Integer enemyUnitId : this.getEnemyUnitIds())
	    	{
	    		Matrix features = this.calculateFeatureVector(state, history, atkUnitId, enemyUnitId);
	    		double qValue = this.calculateQValue(features);
	
	    		if(qValue > maxQ)
	    		{
	    			maxQ = qValue;
	    			featureVec = features;
	    			tgtUnitId = enemyUnitId;
	    		}
	    	}
    	}

    	// remember the info for this unit
    	this.getOldInfoPerUnit().put(atkUnitId, new Triple<Matrix, Matrix, Double>(featureVec, Matrix.full(1, 1, maxQ), r));

    	return tgtUnitId;
    }

    /**
     * This method calculates what the "true" Q(s,a) value should have been based on the Bellman equation for Q-values
     *
     * @param state The current state of the game
     * @param history The current history of the game
     * @param unitId The friendly unitId under consideration
     * @return
     */
    private Matrix getTDGroundTruth(StateView state, HistoryView history, int unitId) throws Exception
    {
    	if(!this.getOldInfoPerUnit().containsKey(unitId))
    	{
    		throw new Exception("unitId=" + unitId + " does not have an old feature vector...cannot calculate TD ground truth for it");
    	}
    	Triple<Matrix, Matrix, Double> oldInfo = this.getOldInfoPerUnit().get(unitId);
    	Double Rs = oldInfo.getThird();

    	double maxQ = Double.NEGATIVE_INFINITY;

    	// try all the actions (i.e. who to attack) in the current state
    	for(Integer tgtUnitId: this.getEnemyUnitIds())
    	{
    		maxQ = Math.max(maxQ, this.calculateQValue(this.calculateFeatureVector(state, history, unitId, tgtUnitId)));
    	}

    	return Matrix.full(1, 1, Rs + QAgent.GAMMA*maxQ); // output is always a scalar in active learning
    }

    /**
     * Calculate the updated weights for this agent. You should construct a matrix
     * @param r The reward R(s) for the prior state
     * @param state Current state of the game.
     * @param history History of the game up until this point
     * @param unitId The unit under consideration
     */
    private void updateParams(StateView state, HistoryView history, int unitId) throws Exception
    {
    	if(!this.getOldInfoPerUnit().containsKey(unitId))
    	{
    		throw new Exception("unitId=" + unitId + " does not have an old feature vector...cannot update params for it");
    	}
    	Triple<Matrix, Matrix, Double> oldInfo = this.getOldInfoPerUnit().get(unitId);
    	Matrix oldFeatureVector = oldInfo.getFirst();
    	Matrix Qsa = oldInfo.getSecond();

    	// reset the optimizer (i.e. reset gradients)
    	this.getOptimizer().reset();

    	// populate gradients
    	this.getQFunction().backwards(oldFeatureVector, this.getLossFunction().backwards(Qsa, this.getTDGroundTruth(state, history, unitId)));

    	// take a step in the correct direction
    	this.getOptimizer().step();
    }


	@Override
	public Map<Integer, Action> initialStep(StateView state, HistoryView history)
	{
		// find who our unitIDs are
		this.myUnits = new HashSet<Integer>();
		for(Integer unitId: state.getUnitIds(this.getPlayerNumber()))
		{
			UnitView unitView = state.getUnit(unitId);
			// System.out.println("Found new unit for player=" + this.getPlayerNumber() + " of type=" + unitView.getTemplateView().getName().toLowerCase() + " (id=" + unitId + ")");

			this.myUnits.add(unitId);
		}
		this.initialMyUnits = this.myUnits.size();
		
		// find the enemy player
		Set<Integer> playerIds = new HashSet<Integer>();
		for(Integer playerId: state.getPlayerNumbers())
		{
			playerIds.add(playerId);
		}
		if(playerIds.size() != 2)
		{
			System.err.println("QAgent.initialStep [ERROR]: expected two players");
			System.exit(-1);
		}
		playerIds.remove(this.getPlayerNumber());
		this.ENEMY_PLAYER_ID = playerIds.iterator().next(); // get first element

		this.enemyUnits = new HashSet<Integer>();
		for(Integer unitId: state.getUnitIds(this.getEnemyPlayerId()))
		{
			UnitView unitView = state.getUnit(unitId);
			// System.out.println("Found new unit for player=" + this.getEnemyPlayerId() + " of type=" + unitView.getTemplateView().getName().toLowerCase() + " (id=" + unitId + ")");

			this.enemyUnits.add(unitId);
		}
		this.initialEnemyUnits = this.enemyUnits.size();
		return this.middleStep(state, history);
	}

	/**
     * You will need to calculate the reward at each step and update your totals. You will also need to
     * check if an event has occurred. If it has then you will need to update your weights and select a new action.
     *
     * If you are using the footmen vectors you will also need to remove killed units. To do so use the historyView
     * to get a DeathLog. Each DeathLog tells you which player's unit died and the unit ID of the dead unit. To get
     * the deaths from the last turn do something similar to the following snippet. Please be aware that on the first
     * turn you should not call this as you will get nothing back.
     *
     * for(DeathLog deathLog : historyView.getDeathLogs(stateView.getTurnNumber() -1)) {
     *     System.out.println("Player: " + deathLog.getController() + " unit: " + deathLog.getDeadUnitID());
     * }
     *
     * You should also check for completed actions using the history view. Obviously you never want a footman just
     * sitting around doing nothing (the enemy certainly isn't going to stop attacking). So at the minimum you will
     * have an event whenever one your footmen's targets is killed or an action fails. Actions may fail if the target
     * is surrounded or the unit cannot find a path to the unit. To get the action results from the previous turn
     * you can do something similar to the following. Please be aware that on the first turn you should not call this
     *
     * Map<Integer, ActionResult> actionResults = historyView.getCommandFeedback(playernum, stateView.getTurnNumber() - 1);
     * for(ActionResult result : actionResults.values()) {
     *     System.out.println(result.toString());
     * }
     *
     * @return New actions to execute or nothing if an event has not occurred.
     */
	@Override
	public Map<Integer, Action> middleStep(StateView state, HistoryView history)
	{
		Map<Integer, Action> actions = new HashMap<Integer, Action>(this.getMyUnitIds().size());

    	// if this isn't the first turn in the game
    	if(state.getTurnNumber() > 0)
    	{

    		// check death logs and remove dead units
    		//removes all dead units from the set of unitIds
    		for(DeathLog deathLog : history.getDeathLogs(state.getTurnNumber() - 1))
    		{
    			if(deathLog.getController() == this.getPlayerNumber())
    			{
    				this.getMyUnitIds().remove(deathLog.getDeadUnitID());
    			}
    			else if(deathLog.getController() == this.getEnemyPlayerId())
    			{
    				this.getEnemyUnitIds().remove(deathLog.getDeadUnitID());
    			}
    		}
    	}

    	// get the previous action history in the previous step
		Map<Integer, ActionResult> prevUnitActions = history.getCommandFeedback(this.playernum, state.getTurnNumber() - 1);

    	for(Integer unitId : this.getMyUnitIds())
    	{
    		// decide what each unit should do (i.e. attack)

    		// calculate the reward for this unit
    		double reward = this.getRewardForUnit(state, history, unitId);

    		// if we are playing a test episode then add these rewards to the total reward for the test games
    		if(this.numTestEpisodesPlayedInBatch != -1)
    		{
    			this.totalRewards.set(this.totalRewards.size() - 1, 
    				this.totalRewards.get(this.totalRewards.size() - 1) + Math.pow(this.GAMMA, (state.getTurnNumber() - 1)/100) * reward);
    		}
    		
    		//if this unit does not have an action or the action was completed or failed...give a unit an action 
    		if(state.getTurnNumber() == 0 || !prevUnitActions.containsKey(unitId) || 
    				prevUnitActions.get(unitId).getFeedback().equals(ActionFeedback.COMPLETED) ||
    				prevUnitActions.get(unitId).getFeedback().equals(ActionFeedback.FAILED))
    		{
    			if(state.getTurnNumber() > 0)
    			{
    				// we have arrived at a new state for that unit, so time to update some gradients
    				try
    				{
						this.updateParams(state, history, unitId);
					} catch (Exception e)
    				{
						System.err.println("QAgent.middleStep [ERROR]: problem updating gradients for transition on unitId=" + unitId);
						e.printStackTrace();
						System.exit(-1);
					}
    			}
    			int tgtUnitId = this.selectAction(state, history, unitId);
    			actions.put(unitId, Action.createCompoundAttack(unitId, tgtUnitId));
    		}
    	}

    	if(actions.size() > 0)
    	{
    		this.getStreamer().streamMove(actions);
    	}
        return actions;
	}

	@Override
	public void terminalStep(StateView state, HistoryView history)
	{
		if(this.isTrainingEpisode())
		{
			// save the model
			this.getQFunction().save(this.getParamFilePath());

			this.numTrainingEpisodesPlayed += 1;
			if((this.numTrainingEpisodesPlayed % QAgent.NUM_TRAINING_EPISODES_IN_BATCH) == 0)
			{
				this.numTestEpisodesPlayedInBatch = 0;
			}
		} else
		{
			this.numTestEpisodesPlayedInBatch += 1;
			if((this.numTestEpisodesPlayedInBatch % QAgent.NUM_TESTING_EPISODES_IN_BATCH) == 0)
			{
				this.numTestEpisodesPlayedInBatch = -1;
				// calculate the average
				this.getTotalRewards().set(this.getTotalRewards().size()-1,
						this.getTotalRewards().get(this.getTotalRewards().size()-1) / QAgent.NUM_TRAINING_EPISODES_IN_BATCH); 
	
				// print the average test rewards
				this.printTestData(this.getTotalRewards());
	
				if(this.numTrainingEpisodesPlayed == this.NUM_EPISODES_TO_PLAY)
				{
					System.out.println("played all " + this.NUM_EPISODES_TO_PLAY + " games!");
					System.exit(0);
				} else
				{
					this.getTotalRewards().add(0.0);
				}
			}
		}
	}

	/**
     * DO NOT CHANGE THIS!
     *
     * Prints the learning curve data described in the assignment. Do not modify this method.
     *
     * @param averageRewards List of cumulative average rewards from test episodes.
     */
    private void printTestData (List<Double> averageRewards)
    {
        System.out.println("");
        System.out.println("Games Played      Average Cumulative Reward");
        System.out.println("-------------     -------------------------");
        for (int i = 0; i < averageRewards.size(); i++)
        {
            String gamesPlayed = Integer.toString(QAgent.NUM_TRAINING_EPISODES_IN_BATCH*(i+1));
            String averageReward = String.format("%.2f", averageRewards.get(i));

            int numSpaces = "-------------     ".length() - gamesPlayed.length();
            StringBuffer spaceBuffer = new StringBuffer(numSpaces);
            for (int j = 0; j < numSpaces; j++)
            {
                spaceBuffer.append(" ");
            }
            System.out.println(gamesPlayed + spaceBuffer.toString() + averageReward);
        }
        System.out.println("");
    }

	@Override
	public void loadPlayerData(InputStream inStream) {}

	@Override
	public void savePlayerData(OutputStream outStream) {}

}
